import os
import fitz  # PyMuPDF
from openai import OpenAI
import streamlit as st
import tiktoken

# --- アプリの基本設定 ---
st.set_page_config(page_title="論文PDF翻訳＆要約アプリ (ページ単位処理)", page_icon="📄", layout="wide")

# --- OpenAI APIクライアントを初期化する関数 (変更なし) ---
def get_openai_client(api_key):
    if 'openai_client' not in st.session_state or st.session_state.api_key != api_key:
        try:
            st.session_state.openai_client = OpenAI(api_key=api_key)
            st.session_state.api_key = api_key
            st.session_state.openai_client.models.list()
        except Exception as e:
            st.error(f"APIキーが無効か、接続に問題があります: {e}")
            return None
    return st.session_state.openai_client

# --- テキスト処理関数 ---

# (変更点) ページごとのテキストリストを返すように変更
def extract_text_from_pdf_by_page(uploaded_file) -> list[str]:
    """アップロードされたPDFファイルからページごとにテキストを抽出し、文字列のリストとして返す"""
    try:
        file_bytes = uploaded_file.getvalue()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text = [page.get_text("text", sort=True) for page in doc]  # sort=Trueで読み取り順を安定化
        doc.close()
        return pages_text
    except Exception as e:
        st.error(f"PDFテキスト抽出エラー: {e}")
        return []

def get_token_count(text, model):
    try: encoding = tiktoken.encoding_for_model(model)
    except KeyError: encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def split_text_into_chunks(text: str, model: str, max_tokens: int = 2000) -> list:
    """テキストをトークン数に基づいてチャンクに分割する関数 (改善版)"""
    chunks = []
    # まずは改行で分割を試みる
    sentences = text.split('\n')
    current_chunk = ""
    for sentence in sentences:
        if get_token_count(current_chunk + sentence + "\n", model) <= max_tokens:
            current_chunk += sentence + "\n"
        else:
            if current_chunk: chunks.append(current_chunk)
            # 1文自体が長い場合はさらに強制分割
            while get_token_count(sentence, model) > max_tokens:
                cut_off_point = int(len(sentence) * (max_tokens / get_token_count(sentence, model)))
                chunks.append(sentence[:cut_off_point])
                sentence = sentence[cut_off_point:]
            current_chunk = sentence + "\n"
    if current_chunk: chunks.append(current_chunk)
    return chunks

# (変更点) ページ単位で翻訳処理を行う新しい関数
def translate_page_by_page(client: OpenAI, pages_text: list[str], model: str, target_language: str) -> str:
    """テキストのリスト（ページごと）を受け取り、ページ単位で翻訳する"""
    if not pages_text: return ""
    
    all_translated_pages = []
    
    # 全体の進捗バーの準備
    progress_bar = st.progress(0, text="翻訳の準備をしています...")

    for i, page_text in enumerate(pages_text):
        page_num = i + 1
        progress_text = f"📄 ページ {page_num}/{len(pages_text)} の翻訳中..."
        progress_bar.progress(i / len(pages_text), text=progress_text)
        
        # 1ページが空ならスキップ
        if not page_text.strip():
            all_translated_pages.append(f"--- ページ {page_num} (内容は空でした) ---")
            continue

        # 1ページ内のテキストをさらにチャンクに分割
        sub_chunks = split_text_into_chunks(page_text, model)
        translated_page_parts = []

        for chunk in sub_chunks:
            try:
                res = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": f"あなたはプロの翻訳家です。以下のテキストを、意味を正確に保ちながら自然な{target_language}に翻訳してください。フォーマットや改行は可能な限り維持してください。"},
                        {"role": "user", "content": chunk}
                    ],
                    temperature=0.2
                )
                translated_page_parts.append(res.choices[0].message.content)
            except Exception as e:
                st.error(f"ページ {page_num} の翻訳中にエラーが発生しました: {e}")
                translated_page_parts.append(f"--- エラー：この部分の翻訳に失敗しました ---")
        
        # 翻訳されたページのパーツを結合
        translated_page = "\n".join(translated_page_parts)
        all_translated_pages.append(translated_page)

    progress_bar.progress(1.0, text="翻訳完了！")
    progress_bar.empty()

    # 全ての翻訳済みページを、区切り線を入れて結合
    return "\n\n\n".join(f"--- ページ {i+1} ---\n\n{content}" for i, content in enumerate(all_translated_pages))


def summarize_text(client, text, model, custom_prompt):
    # (この関数は前回の段階的要約のままでOK)
    if not text: return ""
    try:
        chunks = split_text_into_chunks(text, model, max_tokens=3000)
        intermediate_summaries = []
        st.info(f"テキストが長いため、{len(chunks)}個のパートに分けて段階的に要約します。")
        bar = st.progress(0, text="段階的要約を開始...")
        for i, chunk in enumerate(chunks):
            prompt = f"以下は論文の一部です。この部分の要点を簡潔にまとめてください。\n\n---\n{chunk}"
            res = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2)
            intermediate_summaries.append(res.choices[0].message.content)
            bar.progress((i + 1) / len(chunks), text=f"パート {i+1}/{len(chunks)} の要約完了")
        combined_summary = "\n".join(intermediate_summaries)
        final_prompt = f"以下は論文の各パートの要約です。これらの情報を使って、ユーザーの指示に従った最終的な要約を作成してください。\n\nユーザーの指示:「{custom_prompt}」\n\n---\n各パートの要約:\n{combined_summary}"
        final_res = client.chat.completions.create(
            model=model, messages=[
                {"role": "system", "content": "あなたは優秀なリサーチアシスタントです。渡された要約の断片を統合し、最終的な回答を生成してください。"},
                {"role": "user", "content": final_prompt}], temperature=0.5)
        bar.empty()
        return final_res.choices[0].message.content
    except Exception as e:
        st.error(f"要約エラー: {e}"); return ""

# --- Streamlit UI部分 (大きな変更はなし) ---
st.title("📄 論文PDF翻訳＆要約アプリ (ページ単位処理)")
st.markdown("PDFを1ページずつ翻訳・要約し、結果をMarkdownファイルとしてダウンロードできます。")

# (サイドバーは変更なし)
with st.sidebar:
    st.header("⚙️ 事前設定")
    api_key = st.text_input("OpenAI APIキー", type="password")
    model_option = st.selectbox("使用するGPTモデル",("gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"))
    st.header("📝 処理内容の選択")
    process_option = st.radio("実行したい処理",('翻訳と要約', '翻訳のみ', '要約のみ'), horizontal=True)
    if '翻訳' in process_option:
        target_language = st.selectbox("翻訳先の言語",("日本語", "英語", "中国語", "韓国語", "ドイツ語", "フランス語"))
    if '要約' in process_option:
        st.subheader("要約の指示")
        summarize_source = st.radio("何を要約しますか？",("元のテキスト", "翻訳後のテキスト"), horizontal=True) if process_option == '翻訳と要約' else "元のテキスト"
        custom_prompt = st.text_area("要約の指示内容","この論文の「背景・目的」「手法」「結果・結論」を明確に分け、日本語で箇条書きでまとめてください。", height=150)

st.header("1. PDFファイルをアップロード")
uploaded_file = st.file_uploader("処理したいPDFを選択", type="pdf")

if 'result_generated' not in st.session_state:
    st.session_state.result_generated = False

if uploaded_file:
    if st.button("実行する", type="primary"):
        if not api_key:
            st.error("サイドバーでOpenAI APIキーを入力してください。"); st.stop()
        
        client = get_openai_client(api_key)
        if not client: st.stop()

        with st.spinner("PDFからテキストをページ毎に抽出中..."):
            # (変更点) ページごとのリストを受け取る
            pages_text_list = extract_text_from_pdf_by_page(uploaded_file)
        
        if not pages_text_list:
            st.error("PDFからテキストを抽出できませんでした。"); st.stop()
        st.success(f"テキスト抽出完了。({len(pages_text_list)}ページ)")

        translated_text, summary_text = "", ""
        
        # (変更点) 新しい翻訳関数を呼び出す
        if '翻訳' in process_option:
            translated_text = translate_page_by_page(client, pages_text_list, model_option, target_language)
        
        if '要約' in process_option:
            # (変更点) 要約のため、ページリストを一つのテキストに結合
            full_original_text = "\n".join(pages_text_list)
            text_for_summary = translated_text if summarize_source == "翻訳後のテキスト" and translated_text else full_original_text
            summary_text = summarize_text(client, text_for_summary, model_option, custom_prompt)
        
        st.success("全ての処理が完了しました！")
        
        st.session_state.summary = summary_text
        st.session_state.translation = translated_text
        st.session_state.original = "\n\n--- Page Break ---\n\n".join(pages_text_list) # 元のテキストもページ区切りで表示
        st.session_state.filename = uploaded_file.name
        st.session_state.result_generated = True
        st.rerun()

# (結果表示部分は変更なし)
if st.session_state.result_generated:
    st.header("📊 処理結果")
    markdown_output = f"# 📄 「{st.session_state.filename}」の処理結果\n\n"
    if st.session_state.summary:
        markdown_output += f"## 概要\n\n{st.session_state.summary}\n\n---\n\n"
    if st.session_state.translation:
        markdown_output += f"## 全文翻訳\n\n{st.session_state.translation}\n\n"
    st.download_button(
        label="✅ 全ての成果をMarkdownファイルでダウンロード",
        data=markdown_output.encode('utf-8'), # utf-8でエンコード
        file_name=f"{os.path.splitext(st.session_state.filename)[0]}_result.md",
        mime="text/markdown",
    )
    tab_titles = []
    if st.session_state.summary: tab_titles.append("要約")
    if st.session_state.translation: tab_titles.append("翻訳結果")
    tab_titles.append("元のテキスト")
    tabs = st.tabs(tab_titles)
    tab_index = 0
    if st.session_state.summary:
        with tabs[tab_index]: st.markdown(st.session_state.summary)
        tab_index += 1
    if st.session_state.translation:
        with tabs[tab_index]: st.text_area(" ", st.session_state.translation, height=500)
        tab_index += 1
    with tabs[tab_index]:
        st.text_area(" ", st.session_state.original, height=500)