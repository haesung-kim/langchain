import getpass
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
# from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
import time
# import pytesseract as tesseract
# import cv2
import urllib.request
import datetime as dt
import sys
import requests
import json

# 명령행 인자로 크롤링할 제품군 받아오기
# ex) MONITOR / LAPTOP / PROJECTOR
'''
if len(sys.argv) != 2:
    print("Please enter product group. ex) 모니터, PC, 기타")
    sys.exit()

product_type = sys.argv[1]
'''

dateInfo = dt.datetime.now()
todayInfo = str(dateInfo.year) + str(dateInfo.month) + str(dateInfo.day)
print("Date : " + todayInfo)
# options = webdriver.ChromeOptions()
chrome_options = webdriver.ChromeOptions()


# 리눅스 환경에서는 아래 두 설정 필요
# 아래 두 줄 주석 처리하면 crontab으로 실행 시 Error 발생함.
# chrome_options.add_argument('--headless')
# chrome_options.add_argument("--no-sandbox")
# 창을 열지 않고 실행할거면 아래 주석 처리
chrome_options.add_argument('window-size=1920, 1080')

# 안전하지 않은 페이지라는 경고 페이지가 뜸..
# Skip하는 option 추가
chrome_options.add_argument("--ignore-certificate-errors")

# default 다운로드 경로 변경
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": r"/home/bsadmin/datas/voc/downloads",
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})

#driver = webdriver.Chrome(options=chrome_options, service=ChromeDriverManager().install())
driver = webdriver.Chrome(options=chrome_options)
ep_id = 'bsdx.lee'
ep_pw = 'bshawkeye0'
ep_otp = '230825'

driver.get('http://sso.lge.com')
print("Start log-in to EP")

# EP 화면에서 id 입력하는 form의 id 값이 USER
time.sleep(3)
driver.find_element(By.ID, "USER").send_keys(ep_id)
time.sleep(1)
driver.find_element(By.ID, "LDAPPASSWORD").send_keys(ep_pw)
time.sleep(1)
driver.find_element(By.ID, "OTPPASSWORD").send_keys(ep_otp)
time.sleep(1)
driver.find_element(By.ID, "loginSsobtn").click()
time.sleep(5)
print("End log-in to EP")

driver.execute_script('window.open("about:blank", "_blank");')

driver.switch_to.window(driver.window_handles[1])

# Intellytics 사이트 접속
driver.get('http://intellytics.lge.com')
print("Enter Intellytic System")
time.sleep(10)


# Home -> Online Real-time Monitoring -> Online Negative Monitoring(Korea)
# 2. Online VOC Raw Data Download (GLOBAL Only) : 긍정/부정 안나뉜 원천 data

# 일단 2번 routine으로 크롤링!

# Home에서 온라인 Real-time Monitoring 선택
# https://intellytics.lge.com/ko-KR/app/Crawling_Intellytics_VOC/Crawling_Online_Realtime_Monitoring_Introduction_KR_Web
# id : renewal_home_dashboard
iframe = driver.find_element(By.ID, 'renewal_home_dashboard')
driver.switch_to.frame(iframe)
driver.find_element(By.XPATH, '/html/body/div/div[3]/div[2]/div[2]/div/div[1]/h4/a').click()


print("Move to Online Real-time Monitoring")
driver.switch_to.window(driver.window_handles[2])
driver.switch_to.default_content()
time.sleep(5)

# Online VOC analysis : /html/body/header/div/div[3]/div[1]/div/div[4]/a/span
driver.find_element(By.XPATH, "/html/body/header/div/div[3]/div[1]/div/div[2]/a/span").click()
time.sleep(2)

print("Move to Online Real-time Monitorin -> Online Negative Monitoring(Korea)")
# /html/body/header/div/div[3]/div[1]/div/div[4]/div/div[2]/ul/li[4]/a
driver.find_element(By.XPATH, "/html/body/header/div/div[3]/div[1]/div/div[2]/div/div[2]/ul/li[3]/a").click()
time.sleep(60)

# Raw Data Page로 이동
# iframe 내부에 있음
iframe = driver.find_element(By.ID, 'ix_iframe')
driver.switch_to.frame(iframe)
time.sleep(2)

# API 주소 (Online Realtime Monitoring - Korea Neagtive)
url = "https://intellyticshome.lge.com:8443/internal/query/apps/intellytics_voc_analysis/OnlineMonitoringKoreaNegative"

# TODO: 요청 데이터 조건 수정하기
# 요청 데이터 조건
payload = {
    "params": {
        "fileName": "download",
        "_.key": 0.01706667027998,
        "business_hq": "",
        "brand": "LG",
        "_time": {
            "startTime": "-7d@h",
            "endTime": "now",
            "label": "최근 7일",
            "_startTime": 1706065200,
            "_endTime": 1706672972
        },
        "factor": "",
        "symptom": "",
        "startSelTime": "",
        "endSelTime": "",
        "_timeZone": "+09:00"
    },
    "queries": [
        {
            "query": "download_post_info",
            "to": "ds_download_post_info"
        }
    ]
}

# 쿠키
cookies = {cookie["name"]: cookie["value"] for cookie in driver.get_cookies()}
cookie_string = "; ".join([f'{x}={y}' for x,y in cookies.items()]) # JSON 형식의 쿠키를 쿠키 문자열로 변환

# 헤더
headers = {
    "Cookie": cookie_string
}

# POST 요청 보내기
response = requests.post(url, json=payload, headers=headers, verify=False)

# TODO: 엑셀 저장 작성하기

# 응답 확인
if response.status_code == 200:
    print(response.json()['data'])
    print("응답을 성공적으로 받았습니다.")
else:
    print("요청에 실패했습니다. 응답 코드:", response.status_code)