from flask import Flask,render_template
from datetime import datetime
from flask import jsonify
from flask import request, abort
from linebot import (
LineBotApi, WebhookHandler
)
from linebot.exceptions import (
InvalidSignatureError
)
from linebot.models import (
MessageEvent, TextMessage, TextSendMessage,
)
import requests
import json
import re
import hmac
import hashlib
import base64

CHANNEL_ACCESS_TOKEN = 'WHpqwvRl/QLEEB62daTZ6OU/ISOicxZSWKeAHIKJrTdjJlzpWd8dyoynlza7JlmtMVpbLrFYjT2qLnQfJ6I/P7Tt8depFx3y6xXdB1ze9+NJm9mrvzhkwe6l0YFChLJ1sfG+1/QSmjjfcmgqJgaDAAdB04t89/1O/w1cDnyilFU='
CHANNEL_SECRET = '7dc1a7cc327ba211589428f743d3ef37'
LINE_ENDPOINT = 'https://api.line.me/v2/bot'

def post(reply_token, messages):
        header = {
                "Content-Type": "application/json",
                "Authorization": "Bearer {}".format(CHANNEL_ACCESS_TOKEN)
        }
        payload = {
                "replyToken": reply_token,
                "messages": messages,
        }
        requests.post(LINE_ENDPOINT+'/message/reply', headers=header, data=json.dumps(payload))

def get_profile(user_id):
        header = {
                "Content-Type": "application/json",
                "Authorization": "Bearer {}".format(CHANNEL_ACCESS_TOKEN)
        }
        return json.loads(requests.get(LINE_ENDPOINT+'/profile/{}'.format(user_id), headers=header, data='{}').text)

def valdation_signature(signature, body):
        if isinstance(body, str) != True:
                body = body.encode()
        gen_signature = hmac.new(CHANNEL_SECRET.encode(), body.encode(), hashlib.sha256).digest()
        gen_signature = base64.b64encode(gen_signature).decode()

        if gen_signature == signature:
                return True
        else:
                return False

app = Flask(__name__)

line_bot_api = LineBotApi('YOUR_CHANNEL_ACCESS_TOKEN')
handler = WebhookHandler('YOUR_CHANNEL_SECRET')

@app.route('/')
def hello_world():
    return render_template('hello.html' ,title="this is Title!",)

@app.route("/index") #アプリケーション/indexにアクセスが合った場合
def index():
   return render_template('index.html')

@app.route("/select")
def select():
  member_dic = {}
  week_list = ['w1', 'w2', 'w3', 'w4']
  month_list = ['Jan', 'Feb', 'Mar']
  year_list = ['2016', '2017', '2018']
  member_dic['week_list'] = week_list
  member_dic['month_list'] = month_list
  member_dic['year_list'] = year_list
  return render_template('select.html', message=member_dic)

@app.route("/callback", methods=['POST'])
def callback():
        if valdation_signature(request.headers.get('X-Line-Signature', ''), request.data.decode()) == False:
                return 'Error: Signature', 403
        app.logger.info('CALLBACK: {}'.format(request.data))

        for event in request.json['events']:
                # follow
                if event['type'] == 'follow':
                        if event['source']['type'] == 'user':
                                profile = get_profile(event['source']['userId'])
                        messages = [
                                {
                                        'type': 'text',
                                        'text': '追加してくれてありがとう。頑張って予測します！',
                                },
                                {
                                        'type': 'text',
                                        'text': 'これからよろしく！',
                                }
                        ]
                        if 'profile' in locals():
                                messages[0]['text'] = '{}さん\n'.format(profile['displayName'])+messages[0]['text']
                        post(event['replyToken'], messages)
                # Message
                elif event['type'] == 'message':
                    message = event['message']
                    if message['type'] == 'text':
                        messages = [
                                {
                                        'type': 'text',
                                        'text': event['message']['text']+"?　だね、分かった！",
                                },
                                {
                                        'type': 'text',
                                        'text': '多分上がるよ！',
                                        #'text': event['message']['text'],
                                },                                {
                                         "type": "sticker",
                                         "packageId": "1",
                                         "stickerId": "114"
                                },
                                #都度繋げる際にURl変更
                                {"type": "image",
                                "originalContentUrl": "https://b29b2eda.ngrok.io/static/images/figure.jpg",
                                "previewImageUrl": "https://b29b2eda.ngrok.io/static/images/figure.jpg"
                                }
                        ]
                        post(event['replyToken'], messages)
        return '', 200, {}


if __name__ == '__main__':
    app.run(debug=True)
