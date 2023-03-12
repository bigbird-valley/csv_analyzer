import os
import openai
from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
from typing import List
from IPython.display import display, Markdown, Latex   
import os
import matplotlib.pyplot as plt
import japanize_matplotlib
import time
import all_func
import csv
import codecs

# openaai api key の取得
from config import open_api_key
openai.api_key = open_api_key

# webアプリの設定
st.set_page_config(
    page_title="試してみた",
    page_icon="🐣",
    layout="wide",
    initial_sidebar_state="auto")

top_image = Image.open('static/img-x2b30NnXY0l49c1BaufFJWhc.png')
main_image = Image.open('static/Bird_penguin_real_4k_cyber_punk_tokyo_night.png')

st.title("ChatGPT CSV 処理アプリ")
st.image(main_image)

## 使用する関数
def get_csv_head(file_path):
    """
    指定されたCSVファイルの先頭10行を取得して文字列として返す関数。

    Args:
        file_path (str): CSVファイルのパス。

    Returns:
        str: CSVファイルの先頭10行をカンマ区切りで連結した文字列。
    """

    reader = csv.reader(codecs.iterdecode(uploaded_file, 'utf-8'))

    text = ''
    for i, row in enumerate(reader):
        if i < 10:
            text += ','.join(row) + '\n'
        else:
            break
    st.write(text)
    # dataframe でも読み込み  

    return text


def completion(new_message_text:str, settings_text:str = '', past_messages:list = []):
    """
    この関数は、OpenAIのChatGPT API(gpt-3.5-turbo)を使用して、新しいメッセージテキスト、オプションの設定テキスト、
    過去のメッセージのリストを入力として受け取り、レスポンスメッセージを生成するために使用されます。

    Args:
    new_message_text (str): モデルがレスポンスメッセージを生成するために使用する新しいメッセージテキスト。
    settings_text (str, optional): 過去のメッセージリストにシステムメッセージとして追加されるオプションの設定テキスト。デフォルトは''です。
    past_messages (list, optional): モデルがレスポンスメッセージを生成するために使用するオプションの過去のメッセージのリスト。デフォルトは[]です。

    Returns:
    tuple: レスポンスメッセージテキストと、新しいメッセージとレスポンスメッセージを追加した過去のメッセージリストを含むタプル。
    """
    if len(past_messages) == 0 and len(settings_text) != 0:
        system = {"role": "system", "content": settings_text}
        past_messages.append(system)
    new_message = {"role": "user", "content": new_message_text}
    past_messages.append(new_message)

    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=past_messages,
        max_tokens=500
    )
    response_message = {"role": "assistant", "content": result.choices[0].message.content}
    past_messages.append(response_message)
    response_message_text = result.choices[0].message.content

    # Save the dataframe to the data folder
    return response_message_text, past_messages



# アプリ挙動
## ファイルをアップロード
uploaded_file = st.file_uploader("Choose a CSV file、アップロードを待っています。", type="csv")

## ファイルがアップロードされるまで処理はない。
if uploaded_file is not None:
    # pandasデータフレームとして読み込み
    df = pd.read_csv(uploaded_file)
    df.to_csv('data.csv', index=False, header=1)

    st.markdown('## データの確認')
    st.dataframe(df.head(5))

    st.markdown('## 基本統計量の確認')
    st.dataframe(df.describe())

    st.markdown('## 欠損値の確認')
    st.dataframe(df.isnull().sum().sort_values(ascending=False))

    with st.spinner('Loading...概要作るの頑張ってます'):
        # pandasデータフレームとして読み込み
        csv_head = get_csv_head(uploaded_file)

        # プロンプト作成
        prompt_making_summary = f"""{csv_head}

        上記データについて markdown を用いて馴れ馴れしいおじさん風にまとめてください。"""

        # API 利用
        summarize_text, past = completion(prompt_making_summary, '', [])
        st.markdown(summarize_text)
    st.success('概要作成!!完了!!')

    with st.spinner('Loading...グラフ作るの頑張ってます'):

        st.markdown('## ChatGPT 生成コードの表示')
        # プロンプト作成
        prompt_making_graph = f"""
        回答は必ずコードだけ出力してください。
        コードは streamlit で表示します。
        df = pd.read_csv('data.csv') を使ってください。
        df を解析してグラフを1つ表示してください、df のカラム名は {df.columns} です。
        表示の際は、st.pyplot(fig) を使ってください
        データの読み込みから、表示は全て Streamlit で行います。
        コメントは # で必ずコメントアウトしてください"""
        
        # API 利用
        graph_text, past = completion(prompt_making_graph, '', [])
        graph_eval_text = graph_text.replace('import matplotlib.pyplot as plt', 'import matplotlib.pyplot as plt\nimport japanize_matplotlib')
        graph_eval_text = graph_text.replace('```python', '')

        try:
            st.code(graph_eval_text)
            st.markdown('## グラフの表示')
            exec(graph_eval_text)
        except:
            st.markdown('## グラフの表示は失敗')
            st.write('残念ですが、ChatGPT の出力に誤りがあったためエラーとなりました。5 回に一回ぐらいうまくグラフが表示できます。')   
            print('error')
    #exec("""os.remove('data/data.csv')"""
    st.success('Data successfully loaded!')