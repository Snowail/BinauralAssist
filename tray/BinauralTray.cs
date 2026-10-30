using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;
using System.Windows.Forms;
using Microsoft.Win32;

public class TraySettings
{
    public int Port { get; set; }
    public bool AllowLan { get; set; }
    public bool GameAutoStart { get; set; }
    public bool StartOnBoot { get; set; }
    public bool TrayHintShown { get; set; }
    public string LastConfig { get; set; }

    public TraySettings()
    {
        Port = 8765;
        AllowLan = true;
        GameAutoStart = true;
        StartOnBoot = false;
        TrayHintShown = false;
        LastConfig = "current";
    }
}

public class ConfigItem
{
    public string Name { get; set; }
    public string Label { get; set; }
    public string Desc { get; set; }
    public bool Builtin { get; set; }
}

public class ConfigResponse
{
    public List<ConfigItem> Items { get; set; }
    public string Last { get; set; }
    public string Current { get; set; }
}

public class StatusResponse
{
    public bool running { get; set; }
    public string error { get; set; }
}

public class SimpleResult
{
    public bool ok { get; set; }
    public string error { get; set; }
    public bool running { get; set; }
}

public class TrayApp : ApplicationContext
{
    private readonly string baseDir;
    private readonly string settingsPath;
    private readonly JavaScriptSerializer json = new JavaScriptSerializer();
    private readonly object logLock = new object();
    private NotifyIcon tray;
    private ContextMenuStrip menu;
    private System.Windows.Forms.Timer timer;
    private Process server;
    private volatile bool webUIExited;
    private bool restarting;
    private bool quitting;
    private TraySettings settings;
    private Icon trayIcon;

    private const string StartupKey = @"Software\Microsoft\Windows\CurrentVersion\Run";
    private const string StartupName = "BinauralAssistTray";

    public TrayApp()
    {
        baseDir = AppDomain.CurrentDomain.BaseDirectory;
        settingsPath = Path.Combine(baseDir, "tray_settings.json");
        settings = LoadSettings();
        settings.StartOnBoot = IsStartupEnabled();

        BuildTray();
        timer = new System.Windows.Forms.Timer();
        timer.Interval = 2000;
        timer.Tick += delegate {
            if (webUIExited) { webUIExited = false; Quit(); }
            else UpdateTrayText();
        };
        timer.Start();

        // WebUI 总会启动；音频由游戏进程监控或用户手动开始。
        StartWebUI();
        SetGameAuto(settings.GameAutoStart);
        UpdateTrayText();
        if (!settings.TrayHintShown)
        {
            tray.ShowBalloonTip(
                5000,
                "Binaural Assist",
                "如果右下角没有图标，请点任务栏的 ^，把 Binaural Assist 拖到任务栏固定。",
                ToolTipIcon.Info);
            settings.TrayHintShown = true;
            SaveSettings();
        }
    }

    private TraySettings LoadSettings()
    {
        try
        {
            if (File.Exists(settingsPath))
            {
                TraySettings value = json.Deserialize<TraySettings>(File.ReadAllText(settingsPath, Encoding.UTF8));
                if (value != null)
                {
                    if (value.Port < 1 || value.Port > 65535) value.Port = 8765;
                    if (string.IsNullOrEmpty(value.LastConfig)) value.LastConfig = "current";
                    return value;
                }
            }
        }
        catch (Exception ex)
        {
            AppendLog("load settings: " + ex.Message);
        }
        return new TraySettings();
    }

    private void SaveSettings()
    {
        try
        {
            File.WriteAllText(settingsPath, json.Serialize(settings), Encoding.UTF8);
        }
        catch (Exception ex)
        {
            AppendLog("save settings: " + ex.Message);
        }
    }

    private void BuildTray()
    {
        menu = new ContextMenuStrip();
        menu.Opening += delegate { BuildMenu(); };
        tray = new NotifyIcon();
        tray.ContextMenuStrip = menu;
        tray.Text = "Binaural Assist";
        trayIcon = LoadTrayIcon();
        tray.Icon = trayIcon;
        tray.Visible = true;
        tray.DoubleClick += delegate { OpenConsole(); };
        BuildMenu();
    }

    private Icon LoadTrayIcon()
    {
        try
        {
            string png = Path.Combine(baseDir, "trayicon.png");
            if (File.Exists(png))
            {
                using (Bitmap bmp = new Bitmap(png))
                {
                    IntPtr handle = bmp.GetHicon();
                    return Icon.FromHandle(handle);
                }
            }
        }
        catch (Exception ex)
        {
            AppendLog("tray icon png: " + ex.Message);
        }
        try
        {
            Icon embedded = Icon.ExtractAssociatedIcon(Application.ExecutablePath);
            if (embedded != null) return embedded;
        }
        catch { }
        return SystemIcons.Application;
    }

    private void BuildMenu()
    {
        bool web = IsFlaskUp();
        bool audio = web && IsAudioRunning();
        menu.Items.Clear();

        ToolStripMenuItem status = new ToolStripMenuItem(
            audio ? "音频处理中" : (web ? "WebUI 已启动 · 音频未启动" : "WebUI 未启动"));
        status.Enabled = false;
        menu.Items.Add(status);
        menu.Items.Add(new ToolStripSeparator());

        ToolStripMenuItem open = new ToolStripMenuItem("打开网页控制台");
        open.Click += delegate { OpenConsole(); };
        menu.Items.Add(open);

        ToolStripMenuItem settingsMenu = new ToolStripMenuItem("设置");
        settingsMenu.DropDownItems.Add(MenuItem("WebUI 监听端口…", delegate { ChangePort(); }));
        settingsMenu.DropDownItems.Add(new ToolStripSeparator());

        ToolStripMenuItem lan = new ToolStripMenuItem("允许局域网控制");
        lan.Checked = settings.AllowLan;
        lan.Click += delegate { ToggleLan(); };
        settingsMenu.DropDownItems.Add(lan);

        ToolStripMenuItem auto = new ToolStripMenuItem("游戏启动自动开启音频处理");
        auto.Checked = settings.GameAutoStart;
        auto.Click += delegate { ToggleGameAuto(); };
        settingsMenu.DropDownItems.Add(auto);

        ToolStripMenuItem boot = new ToolStripMenuItem("开机启动");
        boot.Checked = settings.StartOnBoot;
        boot.Click += delegate { ToggleStartOnBoot(); };
        settingsMenu.DropDownItems.Add(boot);

        settingsMenu.DropDownItems.Add(new ToolStripSeparator());
        settingsMenu.DropDownItems.Add(BuildConfigMenu(web));
        menu.Items.Add(settingsMenu);

        menu.Items.Add(new ToolStripSeparator());
        ToolStripMenuItem quit = new ToolStripMenuItem("关闭服务并退出");
        quit.Click += delegate { Quit(); };
        menu.Items.Add(quit);
    }

    private ToolStripMenuItem BuildConfigMenu(bool running)
    {
        ToolStripMenuItem root = new ToolStripMenuItem("切换当前配置");
        if (!running)
        {
            ToolStripMenuItem none = new ToolStripMenuItem("服务未运行");
            none.Enabled = false;
            root.DropDownItems.Add(none);
            return root;
        }
        try
        {
            ConfigResponse response = json.Deserialize<ConfigResponse>(HttpGet(Url("/api/configs")));
            if (response == null || response.Items == null || response.Items.Count == 0)
            {
                ToolStripMenuItem none = new ToolStripMenuItem("没有配置");
                none.Enabled = false;
                root.DropDownItems.Add(none);
                return root;
            }
            foreach (ConfigItem item in response.Items)
            {
                ConfigItem captured = item;
                ToolStripMenuItem entry = new ToolStripMenuItem(item.Label);
                entry.Checked = item.Name == response.Last;
                entry.Click += delegate { LoadConfig(captured.Name, captured.Label); };
                root.DropDownItems.Add(entry);
            }
        }
        catch (Exception ex)
        {
            AppendLog("config menu: " + ex.Message);
            ToolStripMenuItem none = new ToolStripMenuItem("无法读取配置");
            none.Enabled = false;
            root.DropDownItems.Add(none);
        }
        return root;
    }

    private ToolStripMenuItem MenuItem(string text, EventHandler handler)
    {
        ToolStripMenuItem item = new ToolStripMenuItem(text);
        item.Click += handler;
        return item;
    }

    private void ChangePort()
    {
        string value = Prompt.Show("WebUI 监听端口", "端口（1–65535）", settings.Port.ToString());
        if (value == null) return;
        int port;
        if (!int.TryParse(value.Trim(), out port) || port < 1 || port > 65535)
        {
            MessageBox.Show("端口必须是 1–65535 的整数。", "Binaural Assist", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }
        bool wasWeb = IsFlaskUp();
        bool wasAudio = wasWeb && IsAudioRunning();
        settings.Port = port;
        SaveSettings();
        if (wasWeb) RestartWebUI(wasAudio);
        BuildMenu();
    }

    private void ToggleLan()
    {
        bool wasWeb = IsFlaskUp();
        bool wasAudio = wasWeb && IsAudioRunning();
        settings.AllowLan = !settings.AllowLan;
        SaveSettings();
        if (wasWeb) RestartWebUI(wasAudio);
        BuildMenu();
    }

    private void ToggleGameAuto()
    {
        settings.GameAutoStart = !settings.GameAutoStart;
        SaveSettings();
        SetGameAuto(settings.GameAutoStart);
        BuildMenu();
    }

    private void SetGameAuto(bool enabled)
    {
        try
        {
            HttpPostJson(Url("/api/games/auto"), json.Serialize(new { enabled = enabled }));
        }
        catch (Exception ex)
        {
            AppendLog("set game auto: " + ex.Message);
        }
    }

    private void ToggleStartOnBoot()
    {
        settings.StartOnBoot = !settings.StartOnBoot;
        try
        {
            using (RegistryKey key = Registry.CurrentUser.OpenSubKey(StartupKey, true))
            {
                if (key == null) throw new Exception("无法打开启动项注册表");
                if (settings.StartOnBoot)
                {
                    key.SetValue(StartupName, "\"" + Application.ExecutablePath + "\" --autostart");
                }
                else
                {
                    key.DeleteValue(StartupName, false);
                }
            }
        }
        catch (Exception ex)
        {
            settings.StartOnBoot = !settings.StartOnBoot;
            MessageBox.Show("设置开机启动失败：\n" + ex.Message, "Binaural Assist", MessageBoxButtons.OK, MessageBoxIcon.Warning);
        }
        SaveSettings();
        BuildMenu();
    }

    private void LoadConfig(string name, string label)
    {
        try
        {
            HttpPostJson(Url("/api/configs/load"), json.Serialize(new { name = name }));
            settings.LastConfig = name;
            SaveSettings();
            BuildMenu();
            tray.ShowBalloonTip(1500, "Binaural Assist", "已切换配置：" + label, ToolTipIcon.Info);
        }
        catch (Exception ex)
        {
            MessageBox.Show("切换配置失败：\n" + ex.Message, "Binaural Assist", MessageBoxButtons.OK, MessageBoxIcon.Warning);
        }
    }

    private void OpenConsole()
    {
        try
        {
            // 双击/打开控制台只启动 WebUI，不自动开始音频处理。
            if (!IsFlaskUp()) StartWebUI();
            Process.Start(new ProcessStartInfo
            {
                FileName = Url("/"),
                UseShellExecute = true
            });
        }
        catch (Exception ex)
        {
            AppendLog("open browser: " + ex.Message);
        }
    }

    private string FindPython()
    {
        string[] candidates = new string[]
        {
            Path.Combine(baseDir, "python", "python.exe"),
            Path.Combine(baseDir, ".venv", "Scripts", "python.exe")
        };
        foreach (string candidate in candidates)
        {
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    private bool StartWebUI()
    {
        if (IsFlaskUp())
        {
            UpdateTrayText();
            return true;
        }
        string python = FindPython();
        if (python == null)
        {
            MessageBox.Show("找不到内置 Python 运行时。\n请保持 python 文件夹与启动器在同一目录。",
                "Binaural Assist", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return false;
        }

        try
        {
            string host = settings.AllowLan ? "0.0.0.0" : "127.0.0.1";
            ProcessStartInfo psi = new ProcessStartInfo();
            psi.FileName = python;
            psi.Arguments = "-m binaural_aed.webui --host " + host + " --port " + settings.Port.ToString();
            psi.EnvironmentVariables["BINAURAL_GAME_AUTO"] = settings.GameAutoStart ? "1" : "0";
            psi.WorkingDirectory = baseDir;
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;
            Process p = new Process();
            p.StartInfo = psi;
            p.EnableRaisingEvents = true;
            p.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e)
            {
                if (!string.IsNullOrEmpty(e.Data)) AppendLog("srv: " + e.Data);
            };
            p.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e)
            {
                if (!string.IsNullOrEmpty(e.Data)) AppendLog("srv: " + e.Data);
            };
            p.Exited += delegate
            {
                if (server == p) server = null;
                if (!restarting && !quitting) webUIExited = true;
            };
            p.Start();
            p.BeginOutputReadLine();
            p.BeginErrorReadLine();
            server = p;

            for (int i = 0; i < 40; i++)
            {
                if (IsFlaskUp())
                {
                    UpdateTrayText();
                    return true;
                }
                if (p.HasExited) break;
                Thread.Sleep(250);
            }
            AppendLog("WebUI did not become ready on " + host + ":" + settings.Port);
            return IsFlaskUp();
        }
        catch (Exception ex)
        {
            AppendLog("start WebUI: " + ex.Message);
            MessageBox.Show("启动 WebUI 失败：\n" + ex.Message, "Binaural Assist", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return false;
        }
    }

    private bool StartAudio()
    {
        if (!IsFlaskUp() && !StartWebUI()) return false;
        if (IsAudioRunning())
        {
            UpdateTrayText();
            return true;
        }
        try
        {
            string response = HttpPostJson(Url("/api/autostart"), "{}");
            SimpleResult result = json.Deserialize<SimpleResult>(response);
            if (result == null || !result.ok)
            {
                string error = result == null ? "未知错误" : result.error;
                AppendLog("autostart audio: " + error);
                if (tray != null)
                {
                    tray.ShowBalloonTip(2500, "Binaural Assist",
                        "上次音频设备不可用，请打开控制台手动启动。",
                        ToolTipIcon.Warning);
                }
                UpdateTrayText();
                return false;
            }
            UpdateTrayText();
            return true;
        }
        catch (Exception ex)
        {
            AppendLog("autostart audio: " + ex.Message);
            if (tray != null)
            {
                tray.ShowBalloonTip(2500, "Binaural Assist",
                    "自动开始音频处理失败，请打开控制台手动启动。",
                    ToolTipIcon.Warning);
            }
            return false;
        }
    }

    private void StopAll()
    {
        try
        {
            HttpPostJson(Url("/api/shutdown"), "{}");
        }
        catch { }
        for (int i = 0; i < 20; i++)
        {
            if (!IsFlaskUp()) break;
            Thread.Sleep(150);
        }
        try
        {
            if (server != null && !server.HasExited) server.Kill();
        }
        catch { }
        server = null;
        UpdateTrayText();
    }

    private void RestartWebUI(bool wasAudio)
    {
        restarting = true;
        try
        {
            StopAll();
            StartWebUI();
            if (wasAudio) StartAudio();
        }
        finally
        {
            restarting = false;
        }
    }

    private void Quit()
    {
        quitting = true;
        timer.Stop();
        StopAll();
        if (tray != null)
        {
            tray.Visible = false;
            tray.Dispose();
        }
        Application.Exit();
    }

    private void UpdateTrayText()
    {
        if (tray == null) return;
        bool audio = IsAudioRunning();
        tray.Text = audio ? "Binaural Assist · 音频处理中" : "Binaural Assist · 音频未启动";
    }

    private string Url(string path)
    {
        return "http://127.0.0.1:" + settings.Port.ToString() + path;
    }

    private bool IsFlaskUp()
    {
        try
        {
            HttpWebRequest request = (HttpWebRequest)WebRequest.Create(Url("/api/status"));
            request.Method = "GET";
            request.Timeout = 900;
            request.ReadWriteTimeout = 900;
            using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
            {
                return (int)response.StatusCode == 200;
            }
        }
        catch
        {
            return false;
        }
    }

    private bool IsAudioRunning()
    {
        if (!IsFlaskUp()) return false;
        try
        {
            StatusResponse status = json.Deserialize<StatusResponse>(HttpGet(Url("/api/status")));
            return status != null && status.running;
        }
        catch
        {
            return false;
        }
    }

    private string HttpGet(string url)
    {
        HttpWebRequest request = (HttpWebRequest)WebRequest.Create(url);
        request.Method = "GET";
        request.Timeout = 2000;
        request.ReadWriteTimeout = 2000;
        using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
        using (StreamReader reader = new StreamReader(response.GetResponseStream(), Encoding.UTF8))
        {
            return reader.ReadToEnd();
        }
    }

    private string HttpPostJson(string url, string body)
    {
        HttpWebRequest request = (HttpWebRequest)WebRequest.Create(url);
        request.Method = "POST";
        request.ContentType = "application/json; charset=utf-8";
        request.Timeout = 2500;
        request.ReadWriteTimeout = 2500;
        byte[] data = Encoding.UTF8.GetBytes(body);
        request.ContentLength = data.Length;
        using (Stream stream = request.GetRequestStream())
        {
            stream.Write(data, 0, data.Length);
        }
        using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
        using (StreamReader reader = new StreamReader(response.GetResponseStream(), Encoding.UTF8))
        {
            return reader.ReadToEnd();
        }
    }

    private bool IsStartupEnabled()
    {
        try
        {
            using (RegistryKey key = Registry.CurrentUser.OpenSubKey(StartupKey, false))
            {
                return key != null && key.GetValue(StartupName) != null;
            }
        }
        catch
        {
            return false;
        }
    }

    private void AppendLog(string message)
    {
        try
        {
            lock (logLock)
            {
                File.AppendAllText(
                    Path.Combine(baseDir, "tray.log"),
                    DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + " " + message + "\r\n",
                    Encoding.UTF8);
            }
        }
        catch { }
    }
}

public class Prompt : Form
{
    private TextBox box;

    public static string Show(string title, string label, string initial)
    {
        using (Prompt form = new Prompt())
        {
            form.Text = title;
            form.Width = 460;
            form.Height = 175;
            form.FormBorderStyle = FormBorderStyle.FixedDialog;
            form.StartPosition = FormStartPosition.CenterScreen;
            form.MaximizeBox = false;
            form.MinimizeBox = false;
            form.TopMost = true;

            Label text = new Label();
            text.Text = label;
            text.Left = 14;
            text.Top = 14;
            text.Width = 420;
            form.Controls.Add(text);

            TextBox input = new TextBox();
            input.Text = initial;
            input.Left = 14;
            input.Top = 42;
            input.Width = 420;
            form.Controls.Add(input);
            form.box = input;

            Button ok = new Button();
            ok.Text = "确定";
            ok.Left = 270;
            ok.Top = 82;
            ok.Width = 78;
            ok.DialogResult = DialogResult.OK;
            form.Controls.Add(ok);

            Button cancel = new Button();
            cancel.Text = "取消";
            cancel.Left = 356;
            cancel.Top = 82;
            cancel.Width = 78;
            cancel.DialogResult = DialogResult.Cancel;
            form.Controls.Add(cancel);

            form.AcceptButton = ok;
            form.CancelButton = cancel;
            if (form.ShowDialog() != DialogResult.OK) return null;
            return form.box.Text;
        }
    }
}

public static class Program
{
    [STAThread]
    public static void Main(string[] args)
    {
        bool created;
        using (Mutex mutex = new Mutex(true, "BinauralAssistTray", out created))
        {
            if (!created)
            {
                MessageBox.Show(
                    "Binaural Assist 已经在运行。\n\n请在任务栏右下角 ^ 中查看托盘图标，可把它拖到任务栏固定。",
                    "Binaural Assist", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new TrayApp());
        }
    }
}
