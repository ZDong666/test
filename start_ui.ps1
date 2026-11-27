$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$lanpaintPython = 'D:\Study\Postgraduate\GPT\envs\lanpaint\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $lanpaintPython)) {
    throw '未找到 LanPaint Python 环境，请使用已安装 Web 依赖的 Python 执行 python -m app.ui。'
}
& $lanpaintPython -m app.ui
