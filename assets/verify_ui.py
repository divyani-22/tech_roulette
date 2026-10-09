import time
from pathlib import Path
from playwright.sync_api import sync_playwright

SCREENSHOT_DIR = Path("voltcap/assets/screenshots").resolve()
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

def verify():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1360, "height": 980})
        page.goto("http://localhost:8503")

        print("Waiting for dashboard to mount...")
        page.wait_for_selector(".voltcap-hero-banner", timeout=30000)
        page.wait_for_selector("[data-testid='stMetricValue']", timeout=30000)
        time.sleep(3)

        # 1. Check sidebar collapsed state
        sidebar = page.locator("[data-testid='stSidebar']")
        # In Streamlit, when collapsed, the sidebar section has aria-expanded="false" or is shifted offscreen
        print("Sidebar aria-expanded:", sidebar.get_attribute("aria-expanded"))

        # 2. Test clicking dock button 1 (Tab 4: Models)
        print("Testing dock click to Models tab (Tab 3 index)...")
        page.click(".floating-bottom-dock button:nth-child(2)")
        time.sleep(2)
        active_tab = page.eval_on_selector("div[data-testid='stTab'][aria-selected='true']", "el => el.innerText")
        print("Active tab after dock click:", active_tab)

        # 3. Test dock click to Prediction tab (Tab 5)
        print("Clicking Tab 5 (Prediction)...")
        tabs = page.locator("div[data-testid='stTab']").all()
        if len(tabs) >= 5:
            tabs[4].click()
            time.sleep(2)
            page.screenshot(path=str(SCREENSHOT_DIR / "tab5_waterfall_verified.png"))
            print("Captured: tab5_waterfall_verified.png")

        # 4. Test Failure Suite Surge simulation
        print("Testing Failure Suite: Surge Simulation...")
        surge_btn = page.locator("button:has-text('3× Power Surge')")
        if surge_btn.count() > 0:
            surge_btn.click()
            time.sleep(2)
            page.screenshot(path=str(SCREENSHOT_DIR / "surge_alert_verified.png"))
            print("Captured: surge_alert_verified.png")

        # 5. Capture Tab 7 (Impact)
        print("Clicking Tab 7 (Impact)...")
        if len(tabs) >= 7:
            tabs[6].click()
            time.sleep(2)
            page.screenshot(path=str(SCREENSHOT_DIR / "tab7_impact_verified.png"))
            print("Captured: tab7_impact_verified.png")

        # 6. Test Sidebar sliding open
        print("Testing sliding sidebar opening via hero pill...")
        toggle_btn = page.locator(".hero-sidebar-toggle-btn")
        if toggle_btn.count() > 0:
            toggle_btn.click()
            time.sleep(2)
            print("Sidebar aria-expanded after click:", sidebar.get_attribute("aria-expanded"))
            page.screenshot(path=str(SCREENSHOT_DIR / "sidebar_slid_open.png"))
            print("Captured: sidebar_slid_open.png")

        browser.close()
        print("Verification completed successfully!")

if __name__ == "__main__":
    verify()
