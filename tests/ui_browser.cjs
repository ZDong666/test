// Optional browser regression check. Start the H5 service first.
// UI_BASE_URL defaults to http://127.0.0.1:7860; requires Playwright + Chromium.
// Uses real preview and empty-mask repair endpoints. No GPU inference is invoked.
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '..');
const artifactDir = path.join(root, '.cache', 'ui-browser');
fs.mkdirSync(artifactDir, { recursive: true });
const errors = [];
const checks = [];
const check = (name) => { checks.push(name); console.log(`PASS ${name}`); };
(async () => {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1080 } });
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.UI_BASE_URL || 'http://127.0.0.1:7860', { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => document.querySelector('#maskCanvas').classList.contains('passive'));
    assert(await page.locator('#repairButton').isDisabled());
    assert(await page.locator('#downloadRow').isHidden());
    await page.screenshot({ path: path.join(artifactDir, 'desktop-empty.png'), fullPage: true });
    check('initial state has no invalid downloads');

    await page.locator('#fileInput').setInputFiles(path.join(root, 'data/samples/original/img0000.png'));
    await page.waitForFunction(() => !document.querySelector('#repairButton').disabled);
    const original = await page.locator('#sourceCanvas').evaluate((canvas) => canvas.toDataURL());
    const canvas = page.locator('#maskCanvas');
    await canvas.scrollIntoViewIfNeeded();
    let box = await canvas.boundingBox();
    await page.mouse.move(box.x + box.width * .35, box.y + box.height * .4);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * .65, box.y + box.height * .6, { steps: 10 });
    await page.mouse.up();
    assert.match(await page.locator('#maskStatus').innerText(), /已标记/);
    const marked = await canvas.evaluate((element) => element.toDataURL());
    assert.equal(await page.locator('#sourceCanvas').evaluate((element) => element.toDataURL()), original);
    await page.locator('#eraseTool').click();
    box = await canvas.boundingBox();
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
    assert.notEqual(await canvas.evaluate((element) => element.toDataURL()), marked);
    await page.locator('#undoMask').click();
    assert.equal(await canvas.evaluate((element) => element.toDataURL()), marked);
    check('brush, eraser and undo preserve the source image');

    let responsePromise = page.waitForResponse((response) => response.url().endsWith('/api/preview'));
    await page.locator('#previewButton').click();
    const previewResponse = await responsePromise;
    assert.equal(previewResponse.status(), 200);
    const brushPreview = await previewResponse.json();
    assert(brushPreview.missing_area > 0 && brushPreview.missing_area < 1);
    await page.waitForFunction(() => !document.querySelector('#previewButton').disabled);
    assert(await page.locator('#resultView').isVisible());
    assert(await page.locator('#resultFigure').isHidden());
    assert(await page.locator('#downloadRow').isHidden());
    check('real brush preview does not expose a fake result');

    await page.locator('.segmented label').filter({ hasText: '上传掩码' }).click();
    assert(await page.locator('#resultView').isHidden());
    assert(await page.locator('#repairButton').isDisabled());
    await page.locator('#maskFileInput').setInputFiles(path.join(root, 'data/masks/img0000_center30_mask.png'));
    await page.waitForFunction(() => !document.querySelector('#repairButton').disabled);
    const overlayAlpha = await canvas.evaluate((element) => {
      const pixels = element.getContext('2d').getImageData(0, 0, 256, 256).data;
      return { outside: pixels[3], inside: pixels[(128 * 256 + 128) * 4 + 3] };
    });
    assert.equal(overlayAlpha.outside, 0);
    assert(overlayAlpha.inside > 0 && overlayAlpha.inside < 255);
    responsePromise = page.waitForResponse((response) => response.url().endsWith('/api/preview'));
    await page.locator('#previewButton').click();
    const uploadPreview = await (await responsePromise).json();
    assert(Math.abs(uploadPreview.missing_area - .3) < .02);
    await page.waitForFunction(() => !document.querySelector('#previewButton').disabled);
    await page.screenshot({ path: path.join(artifactDir, 'desktop-preview.png'), fullPage: true });
    check('uploaded mask keeps unmasked source visible and matches real preview');

    await page.locator('.segmented label').filter({ hasText: '画笔涂抹' }).click();
    await page.locator('#seedInput').fill('1.5');
    await page.locator('#repairButton').click();
    assert.match(await page.locator('#errorBox').innerText(), /整数/);
    await page.locator('#seedInput').fill('42');
    responsePromise = page.waitForResponse((response) => response.url().endsWith('/api/repair'));
    await page.locator('#repairButton').click();
    const repaired = await responsePromise;
    assert.equal(repaired.status(), 200);
    const payload = await repaired.json();
    assert.equal(payload.status.nfe, 0);
    await page.waitForFunction(() => !document.querySelector('#repairButton').disabled);
    assert(await page.locator('#downloadRow').isVisible());
    for (const id of ['resultDownload', 'configDownload']) {
      const href = await page.locator(`#${id}`).getAttribute('href');
      assert.equal((await page.request.get(new URL(href, page.url()).href)).status(), 200);
    }
    await page.locator('#compareTab').click();
    await page.locator('#compareRange').focus();
    await page.keyboard.press('ArrowRight');
    assert.equal(await page.locator('#compareValue').innerText(), '51%');
    assert(await page.locator('#compareView').isVisible());
    check('seed validation, real empty-mask repair, downloads and keyboard comparison');

    await page.locator('#budgetSelect').selectOption('2');
    assert(await page.locator('#resultView').isHidden());
    assert.equal(await page.locator('#resultDownload').getAttribute('href'), null);
    check('changed settings invalidate old output and links');

    let releaseRequest;
    const delayed = new Promise((resolve) => { releaseRequest = resolve; });
    await page.route('**/api/repair', async (route) => {
      await delayed;
      await route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: '测试服务暂不可用，请重试。' }) });
    });
    await page.locator('#repairButton').click();
    await page.waitForFunction(() => document.querySelector('#workspace').getAttribute('aria-busy') === 'true');
    for (const id of ['fileInput', 'seedInput', 'budgetSelect', 'previewButton', 'repairButton']) assert(await page.locator(`#${id}`).isDisabled());
    assert.match(await page.locator('#statusText').innerText(), /正在加载模型/);
    assert(await page.locator('#statusText').isVisible());
    releaseRequest();
    await page.waitForFunction(() => !document.querySelector('#repairButton').disabled);
    assert.match(await page.locator('#errorBox').innerText(), /测试服务暂不可用/);
    await page.unroute('**/api/repair');
    check('delayed error response locks inputs then restores controls (mocked)');

    await page.locator('.segmented label').filter({ hasText: '上传掩码' }).click();
    const whiteMask = await page.evaluate(() => {
      const canvas = document.createElement('canvas'); canvas.width = canvas.height = 256;
      const context = canvas.getContext('2d'); context.fillStyle = '#fff'; context.fillRect(0, 0, 256, 256);
      return canvas.toDataURL().split(',')[1];
    });
    await page.locator('#maskFileInput').setInputFiles({ name: 'all-white.png', mimeType: 'image/png', buffer: Buffer.from(whiteMask, 'base64') });
    await page.waitForFunction(() => !document.querySelector('#previewButton').disabled);
    await page.locator('#previewButton').click();
    assert.match(await page.locator('#errorBox').innerText(), /全部遮挡/);
    check('all-missing mask rejected before submission');

    for (const width of [320, 390, 768, 1024, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `horizontal overflow at ${width}`);
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('.segmented label').filter({ hasText: '画笔涂抹' }).click();
    await page.screenshot({ path: path.join(artifactDir, 'mobile.png'), fullPage: true });
    assert.deepEqual(errors, []);
    check('responsive widths 320–1440 and no browser exceptions');
    fs.writeFileSync(path.join(artifactDir, 'report.json'), JSON.stringify({ checks, errors, gpuInference: false }, null, 2));
  } finally { await browser.close(); }
})().catch((error) => { console.error(error); process.exitCode = 1; });

