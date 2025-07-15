import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# User personal info
APPLICANT_INFO = {
    "first_name": "Dyllun",
    "last_name": "Garrett",
    "dob": "08/08/1994",
    "ssn": "541-45-1328",
    "phone": "503-908-4611",
    "email": "your_email@example.com",  # Replace with actual email
    "address": "12812 Southeast Herald Street, Portland, Oregon 97236",
    "time_at_residence": "5",
    "housing_type": "Rent",
    "monthly_rent": "1200",
    "employer": "Broadcast Music, Inc. (BMI)",
    "occupation": "Cue Sheets Manager",
    "employment_start": "2014-01-01",
    "monthly_income": "4250",
    "employer_phone": "310-659-9109",
    "employer_address": "9420 Wilshire Boulevard, Beverly Hills, CA 90212",
    "license_number": "2744164",
    "license_state": "Oregon",
    "license_exp": "08/08/2026",
    "trade_in_description": "2006 Buell Ulysses XB12X needs front right fork seal, rear wheel bearings, drive belt"
}

DEALERS = {
    "Carter Motorsports": {
        "url": "https://www.cartermotorsports.com/credit-financing-atvs-motorcycles-utvs-dealership--financing"
    },
    "Burnaby Kawasaki": {
        "url": "https://www.burnabykawasaki.com/credit-financing-motorcycles-dealership--financing"
    },
    "International Motorsports": {
        "url": "https://www.internationalmotorsports.com/international-motorsports-financing"
    },
    "Beaverton Motorcycles": {
        "url": "https://app.revvable.com/dealers/103af3a8-2411-423f-be05-58e113c52210/credit/intake?mode=full"
    },
    "Paradise Harley-Davidson": {
        "url": "https://www.paradiseh-d.com/buy-a-harley-davidson-with-credit--financing"
    },
    "Motosport Hillsboro": {
        "url": "https://www.motosporthillsboro.com/honda-kawasaki-suzuki-ktm-buy--financing"
    },
    "Sargent’s Motorsports": {
        "url": "https://www.sargentsmotorsports.com/Services/Secure-Financing"
    }
}

def fill_form(driver, dealer_name, info):
    """Example form filler. Update selectors for actual site."""
    # This needs to be adapted for each website's actual form fields
    try:
        driver.find_element(By.NAME, "first_name").send_keys(info["first_name"])
        driver.find_element(By.NAME, "last_name").send_keys(info["last_name"])
        driver.find_element(By.NAME, "email").send_keys(info["email"])
        driver.find_element(By.NAME, "phone").send_keys(info["phone"])
        # Add more fields as needed
        print(f"Filled form for {dealer_name}")
        time.sleep(2)
    except Exception as e:
        print(f"Failed to fill form for {dealer_name}: {e}")


def main():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)

    for dealer, data in DEALERS.items():
        url = data["url"]
        print(f"Opening {dealer} - {url}")
        driver.get(url)
        time.sleep(3)  # wait for page to load
        fill_form(driver, dealer, APPLICANT_INFO)
        # driver.find_element(By.CSS_SELECTOR, "input[type='submit']").click()  # uncomment when selectors are correct
        print(f"Submitted form for {dealer}")
        time.sleep(2)

    driver.quit()

if __name__ == "__main__":
    main()
