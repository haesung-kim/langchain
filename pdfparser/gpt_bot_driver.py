from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time

class gpt_bot_driver:
    def __init__(self, user_id):
        self.user_id = user_id
        self.ep_url = "http://newep.lge.com/portal/main/portalMain.do#a"
        self.lgini_xpath = '//*[@id="divTopMenu"]/div[2]/ul/li[2]/a'
        self.gptbot_xpath = '//*[@id="talksy-footer"]/div[1]/ul/li[3]'
        self.gpt_is_ready = False
        options = Options()
        options.page_load_strategy = 'normal'
        # options.add_argument('--headless')  # 브라우저 창을 숨기는 옵션 추가
        self.driver = webdriver.Chrome(options=options)

        self.driver.get(self.ep_url)
        self._login()
        self._activate_lgini()
        self._activate_gptbot()
        self._close_unnecessary_windows()

    def _login(self):
        wait = WebDriverWait(self.driver, 30)
        username = wait.until(EC.presence_of_element_located((By.ID, 'USER')))
        username.send_keys(self.user_id)
        self.driver.find_element(By.CLASS_NAME, 'Lbody').click()
        wait.until(EC.presence_of_element_located((By.ID, 'loginBiobtn'))).click()

    def _activate_lgini(self):
        wait = WebDriverWait(self.driver, 30)
        wait.until(EC.presence_of_element_located((By.XPATH, self.lgini_xpath))).click()
        self.driver.switch_to.window(self.driver.window_handles[-1])

    def _activate_gptbot(self):
        wait = WebDriverWait(self.driver, 30)
        wait.until(EC.presence_of_element_located((By.XPATH, self.gptbot_xpath))).click()
        self.driver.switch_to.window(self.driver.window_handles[-1])

    def _close_unnecessary_windows(self):
        current_window_handle = self.driver.current_window_handle
        for window_handle in self.driver.window_handles[:-1]:
            self.driver.switch_to.window(window_handle)
            self.driver.close()
        self.driver.switch_to.window(current_window_handle)
        wait = WebDriverWait(self.driver, 30)
        wait.until(EC.presence_of_element_located((By.XPATH, "//*[@id='inputTextArea']")))
        self.gpt_is_ready = True

    # GPT봇 질의 입력하기
    def get_gpt_answer(self, question_text):
        if self.gpt_is_ready:
            previous_count = len(self.driver.find_elements(By.CSS_SELECTOR, 'div.activity'))
            # 질의 입력
            input_text_area = self.driver.find_element(By.XPATH, "//*[@id='inputTextArea']")
            input_text_area.send_keys(question_text + Keys.ENTER)
            while True:
                # activity 클래스 요소 개수 확인
                activity_elements = self.driver.find_elements(By.CSS_SELECTOR, 'div.activity')
                current_count = len(activity_elements)
                if current_count == previous_count + 2:
                    break
                time.sleep(0.5)
            # 답변 대기
            wait = WebDriverWait(self.driver, 60)
            output_div = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.activity.last div.attach.gptDefault")))
            # 답변 텍스트 리턴
            return output_div.text
        else:
            return "Gpt is not ready, yet."
    
    def delete_conversation_history(self):
        wait = WebDriverWait(self.driver, 60)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li.callApp.deleteActivities"))).click()
        alert = self.driver.switch_to.alert
        alert.accept()
        # wait.until(EC.presence_of_element_located((By.XPATH, "//*[@id='inputTextArea']")))
        while True:
            # activity 클래스 요소 개수 확인
            activity_elements = self.driver.find_elements(By.CSS_SELECTOR, 'div.activity')
            current_count = len(activity_elements)
            if current_count == 0:
                break
            time.sleep(0.1)
        time.sleep(2.0)
        
# 사용 예시
if __name__ == "__main__":
    import time
    
    USER_ID = "haesung.kim"
    bot = gpt_bot_driver(USER_ID)
    # 질문1
    answer = bot.get_gpt_answer("첫번째 질문입니다.")
    print(answer)
    # 대화 삭제
    bot.delete_conversation_history()
    # 질문 2
    answer = bot.get_gpt_answer("첫번째 질문입니다.")
    print(answer)
    time.sleep(60)
    
