import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
import plotly.graph_objects as go
from transformers import pipeline
import time

# --- CẤU HÌNH BAN ĐẦU ---
st.set_page_config(page_title="Trợ lý Cảm xúc 2-trong-1", layout="wide")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords') # Tải stop words cho mô hình Tiếng Anh

#=========================================================
# PHẦN 1: TẢI CÁC BỘ NÃO (MODELS)
#=========================================================

# --- Não 1: PhoBERT (Tiếng Việt) ---
@st.cache_resource
def load_phobert_model():
    """Tải mô hình PhoBERT (Tiếng Việt) từ thư mục LOCAL."""
    model_name = "phobert_model" 
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=model_name,
            use_fast=False
        )
        print("Tải mô hình PhoBERT (Tiếng Việt) thành công!")
        return sentiment_pipeline
    except Exception as e:
        print(f"LỖI KHI TẢI MODEL POBERT LOCAL: {e}")
        return None

# --- Não 2: Logistic Regression (Tiếng Anh) ---
@st.cache_resource
def load_english_model():
    """Tải mô hình Logistic Regression (Tiếng Anh) từ file .pkl"""
    try:
        with open('sentiment_model_english.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('tfidf_vectorizer_english.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        print("Tải mô hình Logistic Regression (Tiếng Anh) thành công!")
        return model, vectorizer
    except FileNotFoundError:
        print("LỖI: Không tìm thấy file .pkl Tiếng Anh.")
        return None, None

#=========================================================
# PHẦN 2: CÁC HÀM XỬ LÝ (CHO CẢ 2 MODELS)
#=========================================================

# --- Hàm cho Não 1 (PhoBERT) ---
def analyze_fragments_vietnamese(text, ai_pipeline):
    """Tách câu Tiếng Việt thành các vế và phân tích."""
    # Logic tách câu đã sửa
    split_words = [',', 'nhưng', 'tuy nhiên', 'tuy vậy', 'dù', 'mặc dù', 'thay vào đó']
    split_pattern = r'(' + ' | '.join(re.escape(word) for word in split_words) + ' )'
    fragments = re.split(split_pattern, text)
    
    final_fragments = []
    temp_frag = ""
    for frag in fragments:
        if frag.strip() in split_words:
            temp_frag = frag + " "
        elif frag.strip():
            final_fragments.append((temp_frag + frag).strip())
            temp_frag = ""
            
    cleaned_fragments = [f for f in final_fragments if f and len(f.split()) > 1] 
    if len(cleaned_fragments) <= 1: return []

    results = []
    for frag in cleaned_fragments:
        result = ai_pipeline(frag)[0]
        results.append((frag, result['label'], result['score']))
    return results

# --- Hàm cho Não 2 (Logistic Regression) ---
def clean_text_english(text):
    """Hàm làm sạch văn bản Tiếng Anh (từ Notebook)"""
    text = str(text).lower() # Chuyển thành chữ thường
    text = re.sub(r'[^\w\s]', '', text) # Xóa ký tự đặc biệt, dấu câu
    text = re.sub(r'\d+', '', text) # Xóa số
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)

#=========================================================
# PHẦN 3: GIAO DIỆN CHÍNH (DÙNG TABS)
#=========================================================

st.title("🚀 Trợ lý Cảm xúc AI 2-trong-1")

# Tải cả 2 mô hình
phobert_pipeline = load_phobert_model()
eng_model, eng_vectorizer = load_english_model()

# Tạo 2 Tabs
tab1, tab2 = st.tabs(["Phân tích Trực tiếp", 
                      "Phân tích File "])

# --- TAB 1: GIAO DIỆN PHOBERT (TIẾNG VIỆT) ---
with tab1:
    if phobert_pipeline is None:
        st.error("Lỗi tải mô hình PhoBERT (Tiếng Việt). Vui lòng kiểm tra thư mục 'phobert_model'.")
    else:
        st.header("Sử dụng Deep Learning (PhoBERT) để bóc tách ngữ cảnh")
        
        col1, col2 = st.columns([0.6, 0.4])
        
        with col1:
            user_input_vi = st.text_area("Nhập bình luận Tiếng Việt:", 
                                         "áo thì đẹp, nhưng phí ship quá đắt thay vào đó nhân viên nhiệt tình", 
                                         height=150, key="vi_input")
            analyze_button_vi = st.button("✨ Phân tích ngay!", key="vi_button")
            
            st.markdown("---")
            st.markdown("#### 💡 Gợi ý thử nghiệm:")
            st.info("Thử nhập một câu có nhiều vế cảm xúc trái ngược nhau, ví dụ:\n\n"
                    "* `Quần áo shop này đẹp, nhưng giá hơi chát.`\n"
                    "* `Dịch vụ tốt, đồ ăn tạm được, sẽ quay lại.`\n"
                    "* `Mặc dù giao hàng chậm, sản phẩm rất tuyệt vời.`")

        with col2:
            st.markdown("### 🔍 Kết quả Phân tích (Tiếng Việt)")
            if analyze_button_vi:
                if not user_input_vi.strip():
                    st.warning("Vui lòng nhập bình luận Tiếng Việt.")
                else:
                    # Phân tích tổng thể
                    st.markdown("#### 1. Cảm xúc tổng thể:")
                    with st.spinner("AI đang suy nghĩ (Tổng thể)..."):
                        time.sleep(0.3)
                        result_vi = phobert_pipeline(user_input_vi)[0]
                        label = result_vi['label']
                        score = result_vi['score']

                    if label == 'POS':
                        st.markdown("<h1 style='text-align: center; font-size: 80px;'>😄</h1>", unsafe_allow_html=True)
                        st.success("TÍCH CỰC (Positive)")
                    elif label == 'NEG':
                        st.markdown("<h1 style='text-align: center; font-size: 80px;'>😡</h1>", unsafe_allow_html=True)
                        st.error("TIÊU CỰC (Negative)")
                    else: # 'NEU'
                        st.markdown("<h1 style='text-align: center; font-size: 80px;'>😐</h1>", unsafe_allow_html=True)
                        st.info("TRUNG LẬP (Neutral)")
                    
                    st.progress(score)
                    st.metric(label="Độ tự tin (Tổng thể):", value=f"{score * 100:.2f} %")

                    # Phân tích bóc tách
                    st.markdown("---")
                    st.markdown("#### 2. Phân tích bóc tách (Chuyên sâu):")
                    with st.spinner("AI đang bóc tách câu..."):
                        fragments_vi = analyze_fragments_vietnamese(user_input_vi, phobert_pipeline)
                    
                    if not fragments_vi:
                        st.write("Câu này đơn giản, không có vế phụ để bóc tách.")
                    else:
                        st.write("AI nhận thấy câu này có nhiều vế cảm xúc:")
                        for frag, label, score in fragments_vi:
                            frag_text = f"**Vế câu:** `\"{frag}\"`"
                            if label == 'POS':
                                st.success(f"{frag_text} ➝ TÍCH CỰC ({score*100:.0f}%)")
                            elif label == 'NEG':
                                st.error(f"{frag_text} ➝ TIÊU CỰC ({score*100:.0f}%)")
                            else:
                                st.info(f"{frag_text} ➝ TRUNG LẬP ({score*100:.0f}%)")

# --- TAB 2: GIAO DIỆN LOGISTIC REGRESSION (TIẾNG ANH) ---
with tab2:
    if eng_model is None or eng_vectorizer is None:
        st.error("Lỗi tải mô hình Tiếng Anh. Vui lòng kiểm tra 2 file .pkl đã được xuất ra từ Notebook.")
    else:
        st.header("Phân tích Cảm xúc File (Logistic Regression - Tiếng Anh)")
        st.write("Tải lên file .csv hoặc .xlsx chứa đánh giá Tiếng Anh của bạn (từ dự án Notebook).")
        
        uploaded_file = st.file_uploader("Chọn file...", type=["csv", "xlsx"], key="eng_uploader")
        
        if uploaded_file:
            # Đọc file
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Lỗi khi đọc file: {e}")
                st.stop()

            st.info(f"Đã tải lên {len(df)} dòng. Vui lòng chọn các cột văn bản (Tiếng Anh).")
            
            # Chọn cột
            available_cols = df.columns.tolist()
            default_cols = [col for col in ['Review', 'Summary', 'text'] if col in available_cols]
            
            col1_index = 0
            if default_cols:
                try:
                    col1_index = available_cols.index(default_cols[0])
                except ValueError:
                    col1_index = 0 

            col2_index = 0
            if len(default_cols) > 1:
                try:
                    col2_index = available_cols.index(default_cols[1]) + 1
                except ValueError:
                    col2_index = 0

            col1 = st.selectbox("Cột 1 (ví dụ: 'Review')", available_cols, index=col1_index, key="col1_eng")
            col2 = st.selectbox("Cột 2 (ví dụ: 'Summary') (Tùy chọn)", [None] + available_cols, index=col2_index, key="col2_eng")

            if st.button("📊 Bắt đầu Phân tích File", key="eng_button"):
                with st.spinner("Đang phân tích file Tiếng Anh..."):
                    # 1. Tạo cột 'text'
                    if col2 and col2 != 'None':
                        df['text_to_analyze'] = df[col1].astype(str).fillna('') + " " + df[col2].astype(str).fillna('')
                    else:
                        df['text_to_analyze'] = df[col1].astype(str).fillna('')

                    # 2. Làm sạch (theo logic Notebook)
                    df['cleaned_text'] = df['text_to_analyze'].apply(clean_text_english)

                    # 3. Vector hóa
                    X_new = eng_vectorizer.transform(df['cleaned_text'])

                    # 4. Dự đoán
                    predictions = eng_model.predict(X_new) # 0 hoặc 1
                    df['Sentiment_Result'] = predictions
                    df['Sentiment_Label'] = df['Sentiment_Result'].map({1: 'Positive', 0: 'Negative'})
                
                st.success("Phân tích file hoàn tất!")
                
                # Hiển thị kết quả
                total_reviews = len(df)
                pos_count = (df['Sentiment_Result'] == 1).sum()
                neg_count = (df['Sentiment_Result'] == 0).sum()

                st.subheader(f"Tổng quan trên {total_reviews} đánh giá (Tiếng Anh):")
                col_metric1, col_metric2 = st.columns(2)
                col_metric1.metric("👍 Positive", f"{pos_count} ({pos_count/total_reviews:.1%})")
                col_metric2.metric("👎 Negative", f"{neg_count} ({neg_count/total_reviews:.1%})")
                
                # === ĐÂY LÀ DÒNG ĐÃ SỬA LỖI ===
                fig = go.Figure(data=[go.Pie(labels=['Positive', 'Negative'],
                                             values=[pos_count, neg_count],
                                             marker={'colors': ['#28a745', '#dc3545']}, # Sửa ở đây
                                             hole=.3)])
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Xem chi tiết dữ liệu đã phân tích (Tiếng Anh)")
                st.dataframe(df)

                # Tải về
                @st.cache_data
                def convert_df(df_to_convert):
                    return df_to_convert.to_csv(index=False).encode('utf-8')

                csv_output = convert_df(df)
                
                st.download_button(label="📥 Tải về kết quả (CSV)", data=csv_output,
                                   file_name="eng_sentiment_results.csv", mime="text/csv")