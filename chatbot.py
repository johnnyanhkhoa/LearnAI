import openai
import speech_recognition as sr

def CallChatbot():
    while True:
        # Tạo một đối tượng Recognizer để quản lý việc ghi âm và nhận dạng giọng nói
        recording = sr.Recognizer()

        # Mở microphone và điều chỉnh cho nhiễu xung quanh
        with sr.Microphone() as source:
            recording.adjust_for_ambient_noise(source)
            print("Please Say something:")
            audio = recording.listen(source)

        try:
            # Sử dụng Google Speech Recognition để nhận dạng văn bản từ âm thanh ghi âm
            recognized_text = recording.recognize_google(audio)
            print("You said: \n" + recognized_text)
        except Exception as e:
            print(e)
            
        # Thay thế 'your_api_key' bằng API Key của bạn
        api_key = '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'

        # Khởi tạo phiên làm việc với API
        openai.api_key = api_key

        # question = str(input('Đặt câu hỏi: '))

        if recognized_text == 'hết' or recognized_text == 'done' or recognized_text == '':
            return False
        else:
            # Gửi yêu cầu cho chatbot
            response = openai.Completion.create(
            engine="text-davinci-002",
            prompt="Hỏi: " + recognized_text + "?",
            max_tokens=500
            )

            # In kết quả
            print(response.choices[0].text)
        
CallChatbot()