import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
import plotly.graph_objects as go
import numpy as np # Cần cho predict_proba

# --- CẤU HÌNH BAN ĐẦU ---
st.set_page_config(page_title="Demo Logistic Regression", layout="wide")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

#=========================================================
# PHẦN 1: TẢI BỘ NÃO (Logistic Regression)
#=========================================================
@st.cache_resource
def load_english_model():
    """Tải mô hình Logistic Regression  từ file .pkl"""
    try:
        with open('sentiment_model_english.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('tfidf_vectorizer_english.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        print("Tải mô hình Logistic Regression  thành công!")
        return model, vectorizer
    except FileNotFoundError:
        print("LỖI: Không tìm thấy file .pkl Tiếng Anh.")
        st.error("LỖI: Không tìm thấy file 'sentiment_model_english.pkl' hoặc 'tfidf_vectorizer_english.pkl'.")
        st.error("Vui lòng đảm bảo bạn đã xuất 2 file .pkl từ Notebook.")
        return None, None

# Tải mô hình
model, vectorizer = load_english_model()

#=========================================================
# PHẦN 2: HÀM XỬ LÝ (Từ Notebook)
#=========================================================
def clean_text_english(text):
    """Hàm làm sạch văn bản (từ Notebook)"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)

#=========================================================
# PHẦN 3: GIAO DIỆN CHÍNH
#=========================================================
st.title("Sentiment Analysis with Logistic Regression")
st.subheader("Dự án Mô hình Học máy - Phân tích Cảm xúc Bình luận Sản phẩm")

if model is None or vectorizer is None:
    st.error("Không thể tải mô hình. Vui lòng kiểm tra file .pkl.")
else:
    # --- TÍNH NĂNG 1: PHÂN TÍCH TRỰC TIẾP ---
    st.markdown("---")
    st.header("1. Phân tích Trực tiếp (Live Analysis)")
    
    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        user_input_eng = st.text_area("Nhập một bình luận :", 
                                     "This product is absolutely fantastic! Highly recommended.", 
                                     height=100, key="eng_input")
        
        analyze_button_live = st.button("✨ Phân tích ngay!", key="live_button")

    with col2:
        st.markdown("### 🔍 Kết quả Phân tích")
        if analyze_button_live:
            if not user_input_eng.strip():
                st.warning("Vui lòng nhập một bình luận .")
            else:
                with st.spinner("Đang xử lý..."):
                    # 1. Làm sạch
                    cleaned_input = clean_text_english(user_input_eng)
                    # 2. Vector hóa
                    vectorized_input = vectorizer.transform([cleaned_input])
                    # 3. Dự đoán
                    prediction = model.predict(vectorized_input)[0] # 0 hoặc 1
                    probability = model.predict_proba(vectorized_input)
                    confidence_score = np.max(probability)
                    
                if prediction == 1:
                    st.markdown("<h1 style='text-align: center; font-size: 80px;'>👍</h1>", unsafe_allow_html=True)
                    st.success("TÍCH CỰC (Positive)")
                else:
                    st.markdown("<h1 style='text-align: center; font-size: 80px;'>👎</h1>", unsafe_allow_html=True)
                    st.error("TIÊU CỰC (Negative)")
                
                st.progress(confidence_score)
                st.metric(label="Độ tự tin của mô hình:", value=f"{confidence_score * 100:.2f} %")


    # --- TÍNH NĂNG 2: PHÂN TÍCH VÙNG DỮ LIỆU (MỚI) ---
    st.markdown("---")
    st.header("2. Phân tích Dữ liệu Dán (Paste-Box Analysis)")
    st.write("Copy và dán nhiều bình luận từ web vào đây, mỗi bình luận 1 dòng.")

    paste_input = st.text_area("Dán các bình luận vào đây:", 
                               """This is a great product!
I hated this, it broke after one day.
Customer service was very helpful.
Not worth the money, sadly.
""", 
                               height=200, key="paste_input")
    
    analyze_button_paste = st.button("🚀 Phân tích Vùng Dữ liệu", key="paste_button")

    if analyze_button_paste:
        if not paste_input.strip():
            st.warning("Vui lòng dán bình luận vào ô.")
        else:
            with st.spinner("Đang phân tích vùng dữ liệu..."):
                # Tách các bình luận ra theo từng dòng
                lines = paste_input.splitlines()
                # Loại bỏ các dòng trống
                reviews = [line.strip() for line in lines if line.strip()]
                
                if not reviews:
                    st.warning("Không tìm thấy bình luận nào.")
                else:
                    # Tạo DataFrame tạm
                    df_paste = pd.DataFrame(reviews, columns=['text_to_analyze'])
                    
                    # Chạy logic y hệt như Phân tích File
                    df_paste['cleaned_text'] = df_paste['text_to_analyze'].apply(clean_text_english)
                    X_new = vectorizer.transform(df_paste['cleaned_text'])
                    predictions = model.predict(X_new)
                    df_paste['Sentiment_Result'] = predictions
                    df_paste['Sentiment_Label'] = df_paste['Sentiment_Result'].map({1: 'Positive', 0: 'Negative'})
                    
                    st.success(f"Phân tích hoàn tất {len(df_paste)} bình luận!")
                    
                    total_reviews = len(df_paste)
                    pos_count = (df_paste['Sentiment_Result'] == 1).sum()
                    neg_count = (df_paste['Sentiment_Result'] == 0).sum()

                    st.subheader(f"Tổng quan {total_reviews} bình luận đã dán:")
                    col_metric1, col_metric2 = st.columns(2)
                    col_metric1.metric("👍 Positive", f"{pos_count} ({pos_count/total_reviews:.1%})")
                    col_metric2.metric("👎 Negative", f"{neg_count} ({neg_count/total_reviews:.1%})")
                    
                    fig = go.Figure(data=[go.Pie(labels=['Positive', 'Negative'],
                                                 values=[pos_count, neg_count],
                                                 marker={'colors': ['#28a745', '#dc3545']},
                                                 hole=.3)])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Chi tiết kết quả:")
                    st.dataframe(df_paste)


    # --- TÍNH NĂNG 3: PHÂN TÍCH HÀNG LOẠT (TỪ FILE) ---
    st.markdown("---")
    st.header("3. Phân tích Hàng loạt (Batch Analysis)")
    st.write("Tải lên file .csv hoặc .xlsx chứa đánh giá  (từ dự án Notebook).")
    
    uploaded_file = st.file_uploader("Chọn file...", type=["csv", "xlsx"], key="eng_uploader")
    
    if uploaded_file:
        # (Toàn bộ code xử lý file giữ nguyên như cũ...)
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")
            st.stop()

        st.info(f"Đã tải lên {len(df)} dòng. Vui lòng chọn các cột văn bản .")
        
        available_cols = df.columns.tolist()
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
            with st.spinner("Đang phân tích file ..."):
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
            
            total_reviews = len(df)
            pos_count = (df['Sentiment_Result'] == 1).sum()
            neg_count = (df['Sentiment_Result'] == 0).sum()

            st.subheader(f"Tổng quan trên {total_reviews} đánh giá :")
            col_metric1, col_metric2 = st.columns(2)
            col_metric1.metric("👍 Positive", f"{pos_count} ({pos_count/total_reviews:.1%})")
            col_metric2.metric("👎 Negative", f"{neg_count} ({neg_count/total_reviews:.1%})")
            
            fig = go.Figure(data=[go.Pie(labels=['Positive', 'Negative'],
                                         values=[pos_count, neg_count],
                                         marker={'colors': ['#28a745', '#dc3545']},
                                         hole=.3)])
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Xem chi tiết dữ liệu đã phân tích ")
            st.dataframe(df)

            @st.cache_data
            def convert_df(df_to_convert):
                return df_to_convert.to_csv(index=False).encode('utf-8')
            csv_output = convert_df(df)
            st.download_button(label="📥 Tải về kết quả (CSV)", data=csv_output,
                               file_name="eng_sentiment_results.csv", mime="text/csv")