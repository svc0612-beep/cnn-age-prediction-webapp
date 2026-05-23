import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import shutil
import tempfile
from PIL import Image
from pathlib import Path

# =========================
# 기본 설정
# =========================
IMG_SIZE = 128
MODEL_PATH = Path("models/best_cnn_age_model.keras")
FACE_MARGIN = 0.25
MATERIALS_DIR = Path("presentation_materials")
METADATA_PATH = Path("metadata_resized_128_filtered.csv")
DEPLOY_SAMPLE_DIR = Path("deploy_samples")
TRAINING_HISTORY_PATH = Path("models/training_history_long.csv")
TEST_METRICS_PATH = Path("models/test_metrics_long.txt")

# =========================
# 모델 불러오기
# =========================
@st.cache_resource
def load_age_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


@st.cache_resource
def load_face_detector():
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))

    if not detector.empty():
        return detector

    # 한글이 포함된 경로에서는 OpenCV가 XML 파일을 못 읽는 경우가 있어 임시 폴더로 복사합니다.
    temp_cascade_path = Path(tempfile.gettempdir()) / "haarcascade_frontalface_default.xml"
    shutil.copyfile(cascade_path, temp_cascade_path)

    detector = cv2.CascadeClassifier(str(temp_cascade_path))

    if detector.empty():
        raise RuntimeError("OpenCV 얼굴 검출 모델을 불러오지 못했습니다.")

    return detector


def crop_face(image):
    """
    촬영 이미지에서 가장 큰 얼굴 영역을 찾아 crop합니다.
    얼굴을 찾지 못하면 None을 반환합니다.
    """
    rgb_image = image.convert("RGB")
    image_array = np.array(rgb_image)

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    detector = load_face_detector()
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return None, None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    margin_x = int(w * FACE_MARGIN)
    margin_y = int(h * FACE_MARGIN)

    x1 = max(x - margin_x, 0)
    y1 = max(y - margin_y, 0)
    x2 = min(x + w + margin_x, image_array.shape[1])
    y2 = min(y + h + margin_y, image_array.shape[0])

    face_array = image_array[y1:y2, x1:x2]
    face_image = Image.fromarray(face_array)

    return face_image, (x1, y1, x2, y2)

# =========================
# 이미지 전처리 함수
# =========================
def preprocess_image(image):
    """
    카메라로 촬영한 이미지를 모델 입력 형태로 변환
    1. RGB 변환
    2. 128x128 resize
    3. numpy 배열 변환
    4. 0~1 정규화
    5. batch 차원 추가
    """
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def prepare_model_image(image):
    image = image.convert("RGB")
    return image.resize((IMG_SIZE, IMG_SIZE))


def save_uploaded_material(uploaded_file):
    MATERIALS_DIR.mkdir(exist_ok=True)

    safe_name = Path(uploaded_file.name).name
    save_path = MATERIALS_DIR / safe_name

    with save_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())

    return save_path


def get_saved_materials():
    if not MATERIALS_DIR.exists():
        return []

    allowed_extensions = {".ppt", ".pptx", ".pdf", ".docx"}
    return sorted(
        [
            path
            for path in MATERIALS_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in allowed_extensions
        ],
        key=lambda path: path.name.lower(),
    )


@st.cache_data
def load_metadata():
    return pd.read_csv(METADATA_PATH)


@st.cache_data
def load_training_history():
    return pd.read_csv(TRAINING_HISTORY_PATH)


def render_visualization_page():
    st.markdown(
        """
        <div class="section-title">데이터 시각화</div>
        <div class="step-card">
            <div class="step-title">학습 이미지 데이터 분석</div>
            <div class="step-desc">
                최종 학습에 사용한 metadata 파일을 기준으로 나이, 성별, 인종 분포와 샘플 이미지를 확인합니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not METADATA_PATH.exists():
        st.error(f"metadata 파일을 찾을 수 없습니다: {METADATA_PATH}")
        return

    df = load_metadata()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("전체 이미지", f"{len(df):,}장")
    col2.metric("최소 나이", f"{int(df['age'].min())}세")
    col3.metric("최대 나이", f"{int(df['age'].max())}세")
    col4.metric("평균 나이", f"{df['age'].mean():.1f}세")

    if TRAINING_HISTORY_PATH.exists():
        history_df = load_training_history().dropna(how="all")
        best_row = history_df.loc[history_df["val_mae"].idxmin()]

        st.markdown("#### 모델 학습 결과")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("최적 epoch", int(best_row["epoch"]) + 1)
        metric_col2.metric("검증 MAE", f"{best_row['val_mae']:.2f}세")

        if TEST_METRICS_PATH.exists():
            test_metrics = {}
            for line in TEST_METRICS_PATH.read_text(encoding="utf-8").splitlines():
                if ": " in line:
                    key, value = line.split(": ", 1)
                    test_metrics[key] = value
            metric_col3.metric("Test MAE", f"{float(test_metrics.get('test_mae', 0)):.2f}세")
        else:
            metric_col3.metric("Test MAE", "약 6.62세")

        chart_df = history_df.set_index(history_df["epoch"] + 1)[["mae", "val_mae"]]
        chart_df.index.name = "epoch"
        st.line_chart(chart_df)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### 나이대 분포")
        age_group_counts = df["age_group"].value_counts().sort_index()
        st.bar_chart(age_group_counts)

    with chart_col2:
        st.markdown("#### Race 분포")
        race_counts = df["race"].value_counts().sort_index()
        st.bar_chart(race_counts)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.markdown("#### Gender 분포")
        gender_counts = df["gender"].value_counts().sort_index()
        st.bar_chart(gender_counts)

    with chart_col4:
        st.markdown("#### 나이별 이미지 수")
        age_counts = df["age"].value_counts().sort_index()
        st.line_chart(age_counts)

    st.markdown("#### 학습 이미지 샘플")
    sample_df = df.sample(n=min(8, len(df)), random_state=42)
    sample_cols = st.columns(4)

    for index, (_, row) in enumerate(sample_df.iterrows()):
        image_path = DEPLOY_SAMPLE_DIR / row["filename"]

        if not image_path.exists():
            image_path = Path(row["filepath"])

        caption = f"{int(row['age'])}세 / gender {row['gender']} / race {row['race']}"

        if image_path.exists():
            with sample_cols[index % 4]:
                st.image(str(image_path), caption=caption, use_container_width=True)


def render_materials_page():
    st.markdown(
        """
        <div class="step-card">
            <div class="step-title">발표 자료 보관함</div>
            <div class="step-desc">
                보고서, PPT, PDF를 업로드해두면 다른 컴퓨터에서도 이 웹앱에 접속해 다운로드할 수 있습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    presentation_file = st.file_uploader(
        "발표 보고서 또는 PPT 파일 업로드",
        type=["ppt", "pptx", "pdf", "docx"],
    )

    if presentation_file is not None:
        saved_path = save_uploaded_material(presentation_file)
        file_size_mb = saved_path.stat().st_size / (1024 * 1024)
        st.success(f"보관 완료: {saved_path.name} ({file_size_mb:.2f} MB)")

    saved_materials = get_saved_materials()

    if saved_materials:
        st.markdown("#### 다운로드 가능한 발표 자료")

        for material_path in saved_materials:
            file_size_mb = material_path.stat().st_size / (1024 * 1024)

            with material_path.open("rb") as file:
                st.download_button(
                    label=f"{material_path.name} 다운로드 ({file_size_mb:.2f} MB)",
                    data=file.read(),
                    file_name=material_path.name,
                    mime="application/octet-stream",
                    use_container_width=True,
                )
    else:
        st.info("아직 업로드된 발표 자료가 없습니다.")

    st.caption(
        "※ 배포 서버가 파일 저장을 유지하는 환경이어야 업로드한 자료가 계속 남습니다. "
        "일부 무료 배포 환경은 서버가 재시작되면 업로드 파일이 사라질 수 있습니다."
    )

# =========================
# 웹 화면 구성
# =========================
st.set_page_config(
    page_title="얼굴 나이 예측 웹앱",
    page_icon="camera",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.05) 25%, transparent 25%) 0 0 / 28px 28px,
            linear-gradient(180deg, #f8fafc 0%, #e8eef5 100%);
        color: #172033;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .hero {
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(300px, 0.75fr);
        gap: 1.5rem;
        align-items: stretch;
        padding: 1.4rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.88);
        box-shadow: 0 18px 55px rgba(15, 23, 42, 0.10);
        margin-bottom: 1.2rem;
    }

    .hero-copy {
        padding: 0.8rem 0.65rem;
    }

    .eyebrow {
        width: fit-content;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: #ccfbf1;
        color: #0f766e;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .hero h1 {
        margin: 0 0 0.65rem;
        color: #0f172a;
        font-size: 2.65rem;
        line-height: 1.2;
        letter-spacing: 0;
    }

    .hero p {
        margin: 0;
        color: #475569;
        font-size: 1.02rem;
        line-height: 1.7;
    }

    .hero-visual {
        border-radius: 8px;
        background: linear-gradient(145deg, #0f172a 0%, #164e63 54%, #0f766e 100%);
        color: white;
        padding: 1.25rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 280px;
        overflow: hidden;
    }

    .visual-label {
        color: #a7f3d0;
        font-size: 0.8rem;
        font-weight: 800;
        margin-bottom: 0.55rem;
    }

    .visual-main {
        font-size: 1.45rem;
        font-weight: 900;
        line-height: 1.25;
        letter-spacing: 0;
    }

    .pipeline-list {
        display: grid;
        gap: 0.55rem;
        margin-top: 1.1rem;
    }

    .pipeline-row {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.65rem 0.75rem;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.14);
    }

    .pipeline-dot {
        width: 0.72rem;
        height: 0.72rem;
        border-radius: 999px;
        background: #facc15;
        flex: 0 0 auto;
    }

    .pipeline-text {
        font-size: 0.9rem;
        font-weight: 750;
    }

    .model-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 1rem 0 1.35rem;
    }

    .model-tile {
        padding: 1rem;
        border-radius: 8px;
        background: #0f172a;
        color: #f8fafc;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.12);
    }

    .model-tile:nth-child(2) {
        background: #155e75;
    }

    .model-tile:nth-child(3) {
        background: #166534;
    }

    .model-tile:nth-child(4) {
        background: #7c2d12;
    }

    .tile-label {
        font-size: 0.78rem;
        font-weight: 800;
        opacity: 0.76;
        margin-bottom: 0.45rem;
    }

    .tile-value {
        font-size: 1.15rem;
        font-weight: 900;
        letter-spacing: 0;
    }

    .notice {
        padding: 1rem 1.1rem;
        border-left: 5px solid #f59e0b;
        border-radius: 8px;
        background: #fffbeb;
        color: #78350f;
        margin: 1rem 0 1.35rem;
        line-height: 1.65;
    }

    .section-title {
        margin: 1.6rem 0 0.75rem;
        color: #0f172a;
        font-size: 1.25rem;
        font-weight: 900;
    }

    .overview-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.8rem 0 1.25rem;
    }

    .overview-card {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(15, 23, 42, 0.08);
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
        min-height: 116px;
    }

    .overview-kicker {
        color: #0f766e;
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .overview-main {
        color: #0f172a;
        font-size: 1.38rem;
        font-weight: 900;
        line-height: 1.2;
        margin-bottom: 0.35rem;
    }

    .overview-sub {
        color: #64748b;
        font-size: 0.86rem;
        line-height: 1.45;
    }

    .flow {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.65rem;
        margin-bottom: 1.25rem;
    }

    .flow-item {
        padding: 0.95rem;
        border-radius: 8px;
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.08);
    }

    .flow-num {
        width: 1.7rem;
        height: 1.7rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: #1d4ed8;
        color: white;
        font-size: 0.82rem;
        font-weight: 900;
        margin-bottom: 0.55rem;
    }

    .flow-title {
        color: #0f172a;
        font-weight: 850;
        margin-bottom: 0.2rem;
    }

    .flow-desc {
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.45;
    }

    .analysis-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 1rem 0;
    }

    .analysis-card {
        padding: 1rem;
        border-radius: 8px;
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
    }

    .analysis-num {
        width: 1.8rem;
        height: 1.8rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: #0f766e;
        color: white;
        font-weight: 900;
        margin-bottom: 0.55rem;
    }

    .analysis-title {
        color: #0f172a;
        font-size: 1rem;
        font-weight: 900;
        margin-bottom: 0.25rem;
    }

    .analysis-desc {
        color: #64748b;
        font-size: 0.86rem;
        line-height: 1.5;
    }

    .analysis-table {
        padding: 1rem 1.1rem;
        border-radius: 8px;
        background: #f8fafc;
        border: 1px solid rgba(15, 23, 42, 0.08);
        margin: 0.8rem 0 1rem;
    }

    .step-card {
        padding: 1.15rem 1.2rem;
        border-radius: 8px;
        border: 1px solid rgba(15, 23, 42, 0.08);
        background: rgba(255, 255, 255, 0.92);
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
        margin-top: 1rem;
        margin-bottom: 0.85rem;
    }

    .step-title {
        color: #0f172a;
        font-size: 1.12rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .step-desc {
        color: #64748b;
        font-size: 0.94rem;
        line-height: 1.55;
    }

    .result-card {
        padding: 1.45rem 1.35rem;
        border-radius: 8px;
        background: linear-gradient(135deg, #0f766e 0%, #1d4ed8 100%);
        color: white;
        box-shadow: 0 16px 44px rgba(15, 118, 110, 0.25);
        margin-top: 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }

    .result-label {
        font-size: 0.95rem;
        opacity: 0.88;
        margin-bottom: 0.35rem;
    }

    .result-age {
        font-size: 2.6rem;
        line-height: 1.1;
        font-weight: 900;
        letter-spacing: 0;
    }

    .status-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }

    .status-pill {
        padding: 0.45rem 0.65rem;
        border-radius: 999px;
        background: #e0f2fe;
        color: #075985;
        font-size: 0.82rem;
        font-weight: 700;
    }

    [data-testid="stSidebar"] {
        background: #0f172a;
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] .sidebar-title {
        color: white;
        font-size: 1.35rem;
        font-weight: 900;
        line-height: 1.25;
        margin-bottom: 0.75rem;
    }

    [data-testid="stSidebar"] .sidebar-box {
        padding: 0.85rem;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.10);
        margin-bottom: 0.75rem;
    }

    [data-testid="stSidebar"] .sidebar-label {
        color: #67e8f9;
        font-size: 0.76rem;
        font-weight: 850;
        margin-bottom: 0.25rem;
    }

    [data-testid="stSidebar"] .sidebar-value {
        color: #f8fafc;
        font-size: 0.95rem;
        font-weight: 760;
    }

    [data-testid="stSidebar"] .nav-link {
        display: block;
        padding: 0.62rem 0.72rem;
        margin-bottom: 0.45rem;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.10);
        color: #f8fafc;
        font-weight: 800;
        text-decoration: none;
    }

    [data-testid="stSidebar"] .nav-link:hover,
    [data-testid="stSidebar"] .nav-link.active {
        border-color: #67e8f9;
        background: rgba(103, 232, 249, 0.16);
        color: #ffffff;
    }

    div[data-testid="stCameraInput"] {
        padding: 1rem;
        border: 1px dashed rgba(15, 23, 42, 0.24);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.72);
    }

    div[data-testid="stCameraInput"] button {
        font-size: 0 !important;
    }

    div[data-testid="stCameraInput"] button::after {
        content: "찰칵";
        font-size: 0.95rem;
        font-weight: 800;
    }

    div[data-testid="stImage"] img {
        border-radius: 8px;
        border: 1px solid rgba(15, 23, 42, 0.08);
    }

    div[data-testid="stFileUploader"] section {
        border-radius: 8px;
        border-color: rgba(15, 23, 42, 0.18);
        background: rgba(255, 255, 255, 0.72);
    }

    .stAlert {
        border-radius: 8px;
    }

    @media (max-width: 760px) {
        .hero {
            padding: 1.45rem;
        }

        .hero h1 {
            font-size: 1.75rem;
        }

        .overview-grid,
        .flow,
        .model-strip,
        .analysis-grid,
        .hero {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <div class="hero-copy">
            <div class="eyebrow">CNN Age Prediction Web App</div>
            <h1>얼굴을 촬영하면 CNN이 나이를 추정합니다</h1>
            <p>
                카메라 입력, 얼굴 검출, 이미지 전처리, CNN 회귀 예측까지 하나의 웹앱에서
                시연할 수 있도록 구성했습니다. 발표 자료도 같은 페이지에서 보관하고 내려받을 수 있습니다.
            </p>
            <div class="status-row">
                <span class="status-pill">직접 학습 CNN</span>
                <span class="status-pill">외부 분석 API 미사용</span>
                <span class="status-pill">얼굴 영역 crop</span>
                <span class="status-pill">발표 자료 보관함</span>
            </div>
        </div>
        <div class="hero-visual">
            <div>
                <div class="visual-label">REAL-TIME PIPELINE</div>
                <div class="visual-main">카메라 사진이 모델 입력으로 바뀌는 과정</div>
                <div class="pipeline-list">
                    <div class="pipeline-row">
                        <span class="pipeline-dot"></span>
                        <span class="pipeline-text">웹캠 이미지 수집</span>
                    </div>
                    <div class="pipeline-row">
                        <span class="pipeline-dot"></span>
                        <span class="pipeline-text">OpenCV 얼굴 검출</span>
                    </div>
                    <div class="pipeline-row">
                        <span class="pipeline-dot"></span>
                        <span class="pipeline-text">128x128 정규화</span>
                    </div>
                    <div class="pipeline-row">
                        <span class="pipeline-dot"></span>
                        <span class="pipeline-text">CNN 나이 회귀 예측</span>
                    </div>
                </div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    page_param = st.query_params.get("page", "predict")
    page_map = {
        "predict": "나이 예측",
        "visualization": "데이터 시각화",
        "materials": "발표 자료",
    }
    selected_page = page_map.get(page_param, "나이 예측")

    predict_active = "active" if selected_page == "나이 예측" else ""
    visualization_active = "active" if selected_page == "데이터 시각화" else ""
    materials_active = "active" if selected_page == "발표 자료" else ""

    st.markdown(
        f"""
        <div class="sidebar-title">프로젝트 대시보드</div>
        <a class="nav-link {predict_active}" href="?page=predict" target="_self">나이 예측</a>
        <a class="nav-link {visualization_active}" href="?page=visualization" target="_self">시각화</a>
        <a class="nav-link {materials_active}" href="?page=materials" target="_self">발표 자료</a>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-box">
            <div class="sidebar-label">MODEL</div>
            <div class="sidebar-value">best_cnn_age_model.keras</div>
        </div>
        <div class="sidebar-box">
            <div class="sidebar-label">DATASET</div>
            <div class="sidebar-value">UTKFace 기반 10,440장</div>
        </div>
        <div class="sidebar-box">
            <div class="sidebar-label">TEST MAE</div>
            <div class="sidebar-value">약 6.62세</div>
        </div>
        <div class="sidebar-box">
            <div class="sidebar-label">POLICY</div>
            <div class="sidebar-value">LLM / 외부 분석 API 미사용</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="notice">
        본 프로젝트는 교육용 모델입니다. 예측 결과는 실제 나이를 보장하지 않으며,
        참고용 추정값입니다.
    </div>
    """,
    unsafe_allow_html=True,
)

if selected_page == "데이터 시각화":
    render_visualization_page()
    st.stop()

if selected_page == "발표 자료":
    render_materials_page()
    st.stop()

st.markdown(
    """
    <div class="model-strip">
        <div class="model-tile">
            <div class="tile-label">데이터 수</div>
            <div class="tile-value">10,440장</div>
        </div>
        <div class="model-tile">
            <div class="tile-label">입력 크기</div>
            <div class="tile-value">128 x 128 x 3</div>
        </div>
        <div class="model-tile">
            <div class="tile-label">모델 출력</div>
            <div class="tile-value">Dense(1)</div>
        </div>
        <div class="model-tile">
            <div class="tile-label">Test MAE</div>
            <div class="tile-value">약 6.62세</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-title">프로젝트 요약</div>
    <div class="overview-grid">
        <div class="overview-card">
            <div class="overview-kicker">DATA</div>
            <div class="overview-main">10,440장</div>
            <div class="overview-sub">UTKFace 기반 10~70세 얼굴 이미지 학습 데이터</div>
        </div>
        <div class="overview-card">
            <div class="overview-kicker">MODEL</div>
            <div class="overview-main">CNN 회귀</div>
            <div class="overview-sub">Dense(1) 출력으로 나이를 숫자 1개로 예측</div>
        </div>
        <div class="overview-card">
            <div class="overview-kicker">INPUT</div>
            <div class="overview-main">128 x 128</div>
            <div class="overview-sub">OpenCV 얼굴 검출 후 crop 이미지를 모델에 입력</div>
        </div>
    </div>

    <div class="section-title">예측 처리 흐름</div>
    <div class="flow">
        <div class="flow-item">
            <div class="flow-num">1</div>
            <div class="flow-title">카메라 촬영</div>
            <div class="flow-desc">웹캠으로 얼굴 사진을 입력합니다.</div>
        </div>
        <div class="flow-item">
            <div class="flow-num">2</div>
            <div class="flow-title">얼굴 검출</div>
            <div class="flow-desc">OpenCV로 얼굴 영역을 찾습니다.</div>
        </div>
        <div class="flow-item">
            <div class="flow-num">3</div>
            <div class="flow-title">전처리</div>
            <div class="flow-desc">128x128 크기와 0~1 값으로 변환합니다.</div>
        </div>
        <div class="flow-item">
            <div class="flow-num">4</div>
            <div class="flow-title">나이 예측</div>
            <div class="flow-desc">직접 학습한 CNN 모델로 결과를 출력합니다.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# 모델 로드
# =========================
if not MODEL_PATH.exists():
    st.error(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    st.stop()

model = load_age_model()

# =========================
# 카메라 입력
# =========================
if "camera_reset_count" not in st.session_state:
    st.session_state.camera_reset_count = 0

st.markdown(
    """
    <div class="step-card">
        <div class="step-title">1. 카메라 촬영</div>
        <div class="step-desc">얼굴이 화면 중앙에 오도록 촬영하면 검출과 예측이 더 안정적입니다.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

camera_image = st.camera_input(
    "카메라로 얼굴 사진을 촬영하세요",
    key=f"camera_input_{st.session_state.camera_reset_count}",
)

# =========================
# 예측 수행
# =========================
if camera_image is not None:
    if st.button("다시 촬영", use_container_width=True):
        st.session_state.camera_reset_count += 1
        st.rerun()

    st.markdown(
        """
        <div class="step-card">
            <div class="step-title">2. 분석 과정</div>
            <div class="step-desc">촬영된 사진이 어떤 과정을 거쳐 모델 입력이 되고, 예측 결과가 나오는지 확인합니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    image = Image.open(camera_image)
    face_image, _ = crop_face(image)

    if face_image is not None:
        input_image = face_image
        detection_message = "OpenCV가 얼굴을 검출하여 얼굴 주변 영역만 crop했습니다."
    else:
        st.warning("얼굴을 찾지 못해 전체 이미지를 사용합니다. 얼굴이 정면에 잘 보이도록 다시 촬영해보세요.")
        input_image = image
        detection_message = "얼굴 검출에 실패하여 전체 촬영 이미지를 모델 입력으로 사용했습니다."

    model_image = prepare_model_image(input_image)

    image_col1, image_col2, image_col3 = st.columns(3)

    with image_col1:
        st.image(image, caption="1. 촬영 원본", use_container_width=True)

    with image_col2:
        st.image(input_image, caption="2. 얼굴 검출/crop 결과", use_container_width=True)

    with image_col3:
        st.image(model_image, caption="3. 모델 입력 이미지 128x128", use_container_width=True)

    input_array = preprocess_image(input_image)

    pred_age = model.predict(input_array, verbose=0)[0][0]

    st.markdown(
        f"""
        <div class="analysis-grid">
            <div class="analysis-card">
                <div class="analysis-num">1</div>
                <div class="analysis-title">사진 입력</div>
                <div class="analysis-desc">웹캠으로 촬영한 이미지를 PIL 이미지로 읽어옵니다.</div>
            </div>
            <div class="analysis-card">
                <div class="analysis-num">2</div>
                <div class="analysis-title">얼굴 영역 선택</div>
                <div class="analysis-desc">{detection_message}</div>
            </div>
            <div class="analysis-card">
                <div class="analysis-num">3</div>
                <div class="analysis-title">모델 전처리</div>
                <div class="analysis-desc">RGB 변환, 128x128 resize, 0~1 정규화, batch 차원 추가를 수행합니다.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="analysis-table">
            <b>모델 입력 정보</b><br>
            원본 이미지 크기: {image.size[0]} x {image.size[1]} px<br>
            모델 입력 크기: {IMG_SIZE} x {IMG_SIZE} x 3<br>
            입력 배열 shape: {input_array.shape}<br>
            픽셀 값 범위: {input_array.min():.3f} ~ {input_array.max():.3f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">CNN 모델 예측 결과</div>
            <div class="result-age">약 {pred_age:.1f}세</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "※ 입력 이미지는 모델 예측을 위해 임시로 처리되며, "
        "현재 코드에서는 별도 DB에 저장하지 않습니다."
    )

else:
    st.warning("카메라로 사진을 촬영하면 예측 결과가 표시됩니다.")
