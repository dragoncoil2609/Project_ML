import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
import plotly.graph_objects as go
import numpy as np
import sqlite3
import bcrypt

# --- CẤU HÌNH BAN ĐẦU ---
st.set_page_config(page_title="Demo Logistic Regression", layout="wide")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

#=========================================================
# PHẦN 1: KẾT NỐI DATABASE VÀ CÁC HÀM XỬ LÝ
#=========================================================
# (Các hàm này giữ nguyên)
def get_db_connection():
    return sqlite3.connect('users.db')

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def save_history(username, analysis_type, input_text, result_label, result_score):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO history (username, analysis_type, input_text, result_label, result_score) VALUES (?, ?, ?, ?, ?)",
        (username, analysis_type, input_text, result_label, result_score)
    )
    conn.commit()
    conn.close()

def get_user_history(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, analysis_type, input_text, result_label, result_score FROM history WHERE username = ? ORDER BY timestamp DESC", (username,))
    history_df = pd.DataFrame(cursor.fetchall(), columns=['Thời gian', 'Loại', 'Input', 'Kết quả', 'Độ tự tin'])
    conn.close()
    return history_df

#=========================================================
# PHẦN 2: TẢI BỘ NÃO AI (Logistic Regression)
#=========================================================
@st.cache_resource
def load_english_model():
    # (Giữ nguyên hàm này)
    try:
        with open('sentiment_model_english.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer_english.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("LỖI: Không tìm thấy file .pkl.")
        return None, None

model, vectorizer = load_english_model()

#=========================================================
# PHẦN 3: HÀM XỬ LÝ VĂN BẢN (Từ Notebook)
#=========================================================
def clean_text_english(text):
    # (Giữ nguyên hàm này)
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)

#=========================================================
# PHẦN 4: GIAO DIỆN CHÍNH (Đăng nhập / Đăng ký TỰ LÀM)
#=========================================================

# Khởi tạo session state
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'name' not in st.session_state:
    st.session_state['name'] = None

# --- Nếu CHƯA ĐĂNG NHẬP ---
if not st.session_state['authentication_status']:
    col_login, col_intro = st.columns([0.5, 0.5]) 

    with col_intro:
        st.title("Sentiment Analysis with Logistic Regression")
        st.markdown("")
        st.markdown("""
        Ứng dụng này sử dụng mô hình **Logistic Regression** đã được huấn luyện trên
        dữ liệu để:
        * Phân tích cảm xúc (Tích cực/Tiêu cực) của một câu.
        * Phân tích hàng loạt bình luận (dán vào hoặc tải file).
        
        Vui lòng **Đăng nhập** hoặc **Đăng ký** (ở bên trái) để bắt đầu.
        """)

    with col_login:
        
        st.subheader("Bảng điều khiển")
        tab_login, tab_register = st.tabs(["🔑 Đăng nhập", "👤 Đăng ký"])

        # --- Tab Đăng nhập ---
        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Tên đăng nhập")
                password = st.text_input("Mật khẩu", type="password")
                login_button = st.form_submit_button("Đăng nhập")

                if login_button:
                    if not (username and password):
                        st.warning("Vui lòng nhập đủ tên đăng nhập và mật khẩu.")
                    else:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("SELECT name, password_hash FROM users WHERE username = ?", (username,))
                        user_data = cursor.fetchone()
                        conn.close()
                        
                        if user_data and check_password(password, user_data[1]):
                            st.session_state['authentication_status'] = True
                            st.session_state['username'] = username
                            st.session_state['name'] = user_data[0]
                            st.rerun() 
                        else:
                            st.error("Tên đăng nhập hoặc mật khẩu không đúng.")

        # --- Tab Đăng ký ---
        with tab_register:
            with st.form("register_form"):
                name = st.text_input("Tên của bạn (ví dụ: 'Hoang Van A')")
                username = st.text_input("Tên đăng nhập (dùng để login)")
                password = st.text_input("Mật khẩu", type="password")
                r_password = st.text_input("Nhập lại Mật khẩu", type="password")
                register_button = st.form_submit_button("Đăng ký")

                if register_button:
                    if not (name and username and password and r_password):
                        st.error("Vui lòng điền đầy đủ thông tin.")
                    elif password != r_password:
                        st.error("Mật khẩu nhập lại không khớp.")
                    else:
                        try:
                            hashed_pass = hash_password(password)
                            conn = get_db_connection()
                            cursor = conn.cursor()
                            cursor.execute("INSERT INTO users (username, password_hash, name) VALUES (?, ?, ?)", (username, hashed_pass, name))
                            conn.commit()
                            conn.close()
                            st.success("Đăng ký thành công! Vui lòng chuyển qua tab 'Đăng nhập'.")
                        except sqlite3.IntegrityError:
                            st.error("Tên đăng nhập này đã tồn tại.")
                        except Exception as e:
                            st.error(f"Lỗi khi đăng ký: {e}")

# --- Nếu ĐÃ ĐĂNG NHẬP ---
else:
    #=========================================================
    # PHẦN 5: GIAO DIỆN ỨNG DỤNG (CHÍNH)
    #=========================================================
    
    # --- THANH SIDEBAR ---
    st.sidebar.title(f"Chào mừng, {st.session_state['name']}!")
    if st.sidebar.button("Đăng xuất"):
        st.session_state['authentication_status'] = None
        st.session_state['username'] = None
        st.session_state['name'] = None
        st.rerun() 

    page = st.sidebar.radio("Điều hướng:", ["Phân tích", "Lịch sử của tôi"])
    st.sidebar.markdown("---")
    
    # --- TRANG "PHÂN TÍCH" ---
    if page == "Phân tích":
        st.title(f"Trang Phân tích")

        # TÍNH NĂNG 1: PHÂN TÍCH TRỰC TIẾP
        st.markdown("---")
        st.header("1. Phân tích Trực tiếp (Live Analysis)")
        # (Code tính năng 1 giữ nguyên)
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            user_input_eng = st.text_area("Nhập một bình luận Tiếng Anh:", "This product is great!", height=100, key="eng_input")
            analyze_button_live = st.button("✨ Phân tích ngay!", key="live_button")
        with col2:
            st.markdown("### 🔍 Kết quả Phân tích")
            if analyze_button_live:
                if user_input_eng.strip() and model:
                    with st.spinner("Đang xử lý..."):
                        cleaned_input = clean_text_english(user_input_eng)
                        vectorized_input = vectorizer.transform([cleaned_input])
                        prediction = model.predict(vectorized_input)[0]
                        probability = model.predict_proba(vectorized_input)
                        confidence_score = np.max(probability)
                        label_text = "Positive" if prediction == 1 else "Negative"
                        
                        if prediction == 1: st.success("TÍCH CỰC (Positive)")
                        else: st.error("TIÊU CỰC (Negative)")
                        st.progress(confidence_score)
                        st.metric(label="Độ tự tin:", value=f"{confidence_score * 100:.2f} %")
                        
                        # Chỉ lưu lịch sử cho tính năng này
                        save_history(st.session_state['username'], "Live", user_input_eng, label_text, confidence_score)

        # TÍNH NĂNG 2: PHÂN TÍCH DỮ LIỆU DÁN
        st.markdown("---")
        st.header("2. Phân tích Dữ liệu Dán (Paste-Box Analysis)")
        # (Code tính năng 2 giữ nguyên)
        paste_input = st.text_area("Dán các bình luận vào đây:", height=200, key="paste_input")
        analyze_button_paste = st.button("🚀 Phân tích Vùng Dữ liệu", key="paste_button")
        if analyze_button_paste:
            if paste_input.strip() and model:
                with st.spinner("Đang phân tích vùng dữ liệu..."):
                    lines = paste_input.splitlines()
                    reviews = [line.strip() for line in lines if line.strip()]
                    if reviews:
                        df_paste = pd.DataFrame(reviews, columns=['text_to_analyze'])
                        df_paste['cleaned_text'] = df_paste['text_to_analyze'].apply(clean_text_english)
                        X_new = vectorizer.transform(df_paste['cleaned_text'])
                        predictions = model.predict(X_new)
                        df_paste['Sentiment_Result'] = predictions
                        df_paste['Sentiment_Label'] = df_paste['Sentiment_Result'].map({1: 'Positive', 0: 'Negative'})
                        
                        st.success(f"Phân tích hoàn tất {len(df_paste)} bình luận!")
                        
                        pos_count = (df_paste['Sentiment_Result'] == 1).sum()
                        neg_count = len(df_paste) - pos_count
                        
                        # === ĐÃ XÓA LỖI LƯU LỊCH SỬ Ở ĐÂY ===
                        
                        total_reviews = len(df_paste)
                        col_metric1, col_metric2 = st.columns(2)
                        col_metric1.metric("👍 Positive", f"{pos_count} ({pos_count/total_reviews:.1%})")
                        col_metric2.metric("👎 Negative", f"{neg_count} ({neg_count/total_reviews:.1%})")
                        fig = go.Figure(data=[go.Pie(labels=['Positive', 'Negative'],
                                                     values=[pos_count, neg_count],
                                                     marker={'colors': ['#28a745', '#dc3545']},
                                                     hole=.3)])
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(df_paste)

        # TÍNH NĂNG 3: PHÂN TÍCH HÀNG LOẠT (FILE)
        st.markdown("---")
        st.header("3. Phân tích Hàng loạt (Batch Analysis)")
        # (Code tính năng 3 giữ nguyên)
        uploaded_file = st.file_uploader("Chọn file...", type=["csv", "xlsx"], key="eng_uploader")
        if uploaded_file and model:
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    header_df = pd.read_excel(uploaded_file, nrows=1)
                else:
                    uploaded_file.seek(0)
                    header_df = pd.read_csv(uploaded_file, nrows=1)
                    uploaded_file.seek(0) 
                    
                available_cols = header_df.columns.tolist()
            except Exception as e:
                st.error(f"Lỗi khi đọc file: {e}")
                available_cols = []
                
            default_cols = [col for col in ['Review', 'Summary', 'text'] if col in available_cols]
            col1_index = 0
            if default_cols:
                try: col1_index = available_cols.index(default_cols[0])
                except ValueError: col1_index = 0 
            col2_index = 0
            if len(default_cols) > 1:
                try: col2_index = available_cols.index(default_cols[1]) + 1
                except ValueError: col2_index = 0
            col1 = st.selectbox("Cột 1 (ví dụ: 'Review')", available_cols, index=col1_index, key="col1_eng")
            col2 = st.selectbox("Cột 2 (ví dụ: 'Summary') (Tùy chọn)", [None] + available_cols, index=col2_index, key="col2_eng")

            
            if st.button("📊 Bắt đầu Phân tích File", key="eng_button"):
                with st.spinner("Đang phân tích file..."):
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                        
                    if col2 and col2 != 'None':
                        df['text_to_analyze'] = df[col1].astype(str).fillna('') + " " + df[col2].astype(str).fillna('')
                    else:
                        df['text_to_analyze'] = df[col1].astype(str).fillna('')
                    
                    df['cleaned_text'] = df['text_to_analyze'].apply(clean_text_english)
                    X_new = vectorizer.transform(df['cleaned_text'])
                    predictions = model.predict(X_new)
                    df['Sentiment_Result'] = predictions
                    df['Sentiment_Label'] = df['Sentiment_Result'].map({1: 'Positive', 0: 'Negative'})
                    
                    st.success("Phân tích file hoàn tất!")
                    
                    pos_count = (df['Sentiment_Result'] == 1).sum()
                    neg_count = len(df) - pos_count
                    
                    # === ĐÃ XÓA LỖI LƯU LỊCH SỬ Ở ĐÂY ===
                    
                    total_reviews = len(df)
                    col_metric1, col_metric2 = st.columns(2)
                    col_metric1.metric("👍 Positive", f"{pos_count} ({pos_count/total_reviews:.1%})")
                    col_metric2.metric("👎 Negative", f"{neg_count} ({neg_count/total_reviews:.1%})")
                    fig = go.Figure(data=[go.Pie(labels=['Positive', 'Negative'],
                                                 values=[pos_count, neg_count],
                                                 marker={'colors': ['#28a745', '#dc3545']},
                                                 hole=.3)])
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df)
                    
                    @st.cache_data
                    def convert_df(df_to_convert):
                        return df_to_convert.to_csv(index=False).encode('utf-8')
                    csv_output = convert_df(df)
                    st.download_button(label="📥 Tải về kết quả (CSV)", data=csv_output,
                                       file_name="eng_sentiment_results.csv", mime="text/csv")


    # --- TRANG "LỊCH SỬ" ---
    elif page == "Lịch sử của tôi":
        st.header(f"Lịch sử Phân tích của {st.session_state['username']}")
        st.write("Đây là các phân tích gần nhất của bạn (từ Phân tích Trực tiếp).")
        
        # Lấy lịch sử từ DB
        history_data = get_user_history(st.session_state['username'])
        
        if history_data.empty:
            st.info("Bạn chưa có lịch sử phân tích nào.")
        else:
            st.dataframe(history_data, use_container_width=True)