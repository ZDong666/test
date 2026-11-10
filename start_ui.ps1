$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$lanpaintPython = 'D:\Study\Postgraduate\GPT\envs\lanpaint\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $lanpaintPython)) {
    throw '未找到 LanPaint Python 环境，请使用已安装依赖的 Python 执行 python -m app.ui。'
}
$env:GRADIO_ANALYTICS_ENABLED = 'False'
& $lanpaintPython -m app.ui
