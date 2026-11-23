"""Deterministic input guards, safety messages, and output hygiene helpers."""
import re
import unicodedata
from urllib.parse import urlsplit

DISCLAIMER = (
    "Bản demo học thuật. Thông tin chỉ để tham khảo và không thay thế tư vấn của nhân viên y tế."
)
PRIVACY = (
    "Đừng nhập họ tên, số điện thoại, địa chỉ, mã hồ sơ y tế hoặc thông tin định danh khác."
)
EMERGENCY = (
    "Đây có thể là tình huống cần hỗ trợ y tế khẩn cấp. "
    "Nếu đang ở Việt Nam, hãy gọi 115 hoặc tìm hỗ trợ cấp cứu trực tiếp ngay."
)
REFUSAL_INSUFFICIENT_EVIDENCE = (
    "Tôi chưa tìm thấy đủ thông tin trong các tài liệu tham khảo để trả lời câu hỏi này "
    "một cách đáng tin cậy. Bạn có thể hỏi cụ thể hơn hoặc tham khảo nhân viên y tế có chuyên môn."
)
REFUSAL = REFUSAL_INSUFFICIENT_EVIDENCE
REFUSAL_OUT_OF_DOMAIN = (
    "Tôi là trợ lý tra cứu kiến thức sức khỏe. "
    "Câu hỏi này nằm ngoài phạm vi tài liệu tôi đang sử dụng."
)
REFUSAL_CONFLICTING_EVIDENCE = (
    "Các tài liệu truy xuất chưa cung cấp thông tin nhất quán để tôi đưa ra câu trả lời "
    "đáng tin cậy. Tôi không muốn suy đoán trong vấn đề sức khỏe."
)
REFUSAL_MISSING_CLINICAL_CONTEXT = (
    "Tôi không có đủ thông tin lâm sàng về tình trạng cụ thể của bạn để đưa ra chẩn đoán, "
    "đơn thuốc hoặc liều dùng an toàn. Vui lòng tham khảo bác sĩ hoặc nhân viên y tế có chuyên môn."
)
PERSONAL = REFUSAL_MISSING_CLINICAL_CONTEXT
GENERATION_UNAVAILABLE = (
    "Hiện tại mô hình tạo câu trả lời chưa sẵn sàng. "
    "Bạn có thể xem các tài liệu đã truy xuất hoặc thử lại khi dịch vụ hoạt động."
)

EMERGENCY_TERMS = (
    "dau nguc", "kho tho", "co giat", "bat tinh", "dot quy",
    "chay mau nhieu", "ngo doc",
    "ngung tho", "ngung tim", "ngat xiu", "bi ngat", "ngat di", "hon me", "mat y thuc",
    "soc phan ve", "suy ho hap", "non ra mau", "ho ra mau",
    "xuat huyet nang", "tu sat", "muon tu tu", "y dinh tu tu", "tu lam hai", "muon tu hai",
)

# Diacritic-sensitive emergency terms matched on NFC-casefolded text so they never
# collide with benign folded phrases like "từ từ", "từ hai", or "ngạt mũi".
ACCENTED_EMERGENCY_TERMS = (
    "tự tử", "tự hại", "ngất",
)

PERSONAL_TERMS = (
    "chan doan cho toi", "chan doan giup toi", "chan doan ho toi",
    "toi bi benh gi", "toi mac benh gi", "toi bi gi", "toi bi sao vay", "toi bi lam sao",
    "toi co bi benh", "ke don", "ke toa", "ke thuoc", "cho toi don thuoc", "don thuoc cho toi",
    "toi nen uong thuoc", "toi nen dung thuoc", "toi can uong thuoc gi", "toi phai uong thuoc gi",
    "toi nen uong bao nhieu", "toi nen dung bao nhieu",
    "uong bao nhieu vien", "uong may vien", "dung may vien",
    "uong bao nhieu ml", "uong bao nhieu giot", "uong bao nhieu mg",
    "lieu bao nhieu", "lieu thuoc cho", "lieu dung cho toi", "lieu dung cho con toi",
    "toi nen uong lieu", "toi nen dung lieu",
    "toi nen uong gi", "nen uong thuoc gi", "nen dung thuoc gi",
    "co nen ngung thuoc", "co nen bo thuoc",
    "thay doi lieu thuoc", "tang lieu thuoc", "giam lieu thuoc",
)

# Diacritic-sensitive health terms that would collide with common non-medical words if folded
# (e.g. "thuốc" vs "thuộc").
ACCENTED_HEALTH_DOMAIN_TERMS = (
    "thuốc", "giấc ngủ", "mất ngủ", "khó ngủ",
)

# Unambiguous normalized cues used ONLY after the dense similarity gate fails,
# to distinguish out-of-domain queries from in-domain queries lacking evidence.
HEALTH_DOMAIN_TERMS = (
    "benh", "trieu chung", "dieu tri", "chua benh", "chua tri", "chua khoi",
    "bac si", "y te", "y khoa", "lam sang", "suc khoe", "benh vien", "kham benh", "di kham",
    "kham suc khoe", "phong kham", "xet nghiem", "xet nghiem mau", "nhom mau",
    "tiem chung", "tiem phong", "tiem vac xin", "chich ngua", "vac xin", "vaccine",
    "huyet ap", "benh tim", "tim mach", "nhip tim", "suy tim", "nhoi mau co tim",
    "benh phoi", "viem phoi", "lao phoi", "ung thu phoi",
    "benh gan", "viem gan", "xo gan", "men gan", "gan nhiem mo", "suy gan",
    "benh than", "suy than", "soi than", "viem cau than", "soi mat", "tui mat",
    "da day", "viem da day", "viem ruot", "ruot thua", "duong ruot",
    "benh nao", "viem nao", "u nao", "nao bo", "tai bien", "dot quy",
    "thieu mau", "mo mau", "mach mau", "chay mau", "dong mau", "duong huyet", "truyen mau",
    "xuong khop", "dau khop", "viem khop", "loang xuong", "gay xuong", "co the",
    "dau dau", "nhuc dau", "dau bung", "dau lung", "dau co", "dau hong", "dau rang",
    "dau mat", "dau da day", "dau vai gay", "dau nhuc", "dau tai", "con dau",
    "bi sot", "ha sot", "sot cao", "sot xuat huyet", "sot virus", "cam sot",
    "ho khan", "ho co dom", "ho keo dai", "tri ho", "giam ho", "con ho",
    "viem", "nhiem trung", "nhiem khuan", "nhiem virus", "lay nhiem", "ung thu",
    "tieu duong", "dai thao duong", "di ung", "hen suyen", "beo phi", "can nang",
    "giam can", "tang can", "sut can", "gout", "gut",
    "mang thai", "thai ky", "thai nhi", "ba bau", "sinh no", "tre so sinh",
    "kinh nguyet", "phu khoa", "nam khoa",
    "dinh duong", "che do an", "an uong", "calo", "vitamin", "ca phe", "caffeine",
    "ruou bia", "thuoc la", "hut thuoc",
    "uong thuoc", "dung thuoc", "loai thuoc", "nha thuoc", "khang sinh", "phau thuat",
    "chan thuong", "vet thuong", "sau rang", "nha khoa", "rang mieng",
    "benh mat", "mo mat", "kho mat", "u tai", "viem tai", "tai mui hong",
    "ngat mui", "so mui", "viem mui", "viem xoang", "viem hong",
    "da lieu", "tam ly", "tram cam", "lo au", "cang thang", "stress",
    "giac ngu", "mat ngu", "kho ngu", "ngu ngon",
    "the duc", "tap the duc", "van dong", "tap luyen",
    "tieu hoa", "ho hap", "than kinh", "noi tiet", "mien dich", "suc de khang",
    "virus", "vi khuan", "ky sinh trung", "covid", "sars", "hiv", "benh lao", "cam cum", "cam lanh",
    "lieu dung", "lieu luong", "lieu thuoc", "qua lieu",
    "tac dung phu", "chong chi dinh", "cap cuu", "duoc pham", "duoc lieu", "duoc si",
    "thuc pham chuc nang", "paracetamol", "ibuprofen", "aspirin", "insulin",
    "cholesterol", "glucose", "bmi", "te bao",
    "chan doan", "tien luong", "phong ngua", "phuc hoi", "hoi chung", "roi loan",
    "ngua da", "man ngua", "noi me day", "phat ban", "sung tay", "sung dau", "sung viem",
    "chong mat", "buon non", "tieu chay", "tao bon", "met moi",
    "sieu am", "noi soi", "sinh thiet", "dich truyen", "sat khuan", "khau trang",
    "u nang", "khoi u", "u xo", "xo vua", "suy nhuoc", "suy dinh duong", "suy giap",
)

OUT_OF_DOMAIN_TERMS = (
    "thu do", "quoc gia", "lich su", "dia ly",
    "song hong", "song me kong", "con song", "ngon nui", "dai duong", "chau luc",
    "lap trinh", "python", "javascript", "java", "code", "phan mem", "tai phan mem",
    "windows", "linux", "macos",
    "thoi tiet", "du bao thoi tiet", "troi nang", "troi mua", "co nang khong", "co mua khong",
    "bong da", "the thao", "world cup",
    "chung khoan", "ty gia", "ngoai te", "bitcoin", "tien ao",
    "phim", "am nhac", "ca si", "dien vien", "nau an", "cong thuc mon",
    "toan hoc", "phuong trinh", "tich phan", "hinh hoc", "du lich", "khach san",
    "chinh tri", "tong thong", "thu tuong", "chien tranh", "hanh tinh", "vu tru", "thien van",
    "xe hoi", "dien thoai", "may tinh", "tro choi", "game",
)

_DOSAGE_RE = re.compile(
    r"\b\d+(?:[,.]\d+)?\s*(?:mg|mcg|[µμ]g|ml|viên|giọt)\b",
    flags=re.IGNORECASE,
)

_PRESCRIPTIVE_PHRASES = (
    "ban nen uong thuoc",
    "ban hay uong thuoc",
    "ban nen dung thuoc",
    "ban hay dung thuoc",
    "toi chan doan ban",
    "chan doan cua ban la",
    "ke don cho ban",
    "don thuoc cua ban",
    "lieu dung cua ban la",
    "ban hay dung lieu",
    "ban nen tang lieu",
    "ban nen giam lieu",
    "ban hay ngung thuoc",
    "ban nen ngung thuoc",
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*\Z", flags=re.IGNORECASE | re.DOTALL)
_STRAY_CLOSE_THINK_RE = re.compile(r"\A.*</think>", flags=re.IGNORECASE | re.DOTALL)
_GEMMA_THINK_BLOCK_RE = re.compile(r"<unused94>\s*thought.*?<unused95>", flags=re.IGNORECASE | re.DOTALL)
_GEMMA_UNCLOSED_THINK_RE = re.compile(r"<unused94>\s*thought.*\Z", flags=re.IGNORECASE | re.DOTALL)
_GEMMA_STRAY_CLOSE_THINK_RE = re.compile(r"\A.*<unused95>", flags=re.IGNORECASE | re.DOTALL)


def _normalize_accented(text):
    if not isinstance(text, str):
        return ""
    folded = unicodedata.normalize("NFC", text.casefold())
    return " ".join(re.findall(r"[^\W_]+", folded, flags=re.UNICODE))


def normalize_vietnamese(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text.casefold()).replace("đ", "d")
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def is_emergency(question):
    if not isinstance(question, str):
        return False
    accented = f" {_normalize_accented(question)} "
    if any(f" {phrase} " in accented for phrase in ACCENTED_EMERGENCY_TERMS):
        return True
    # Exclude benign "ngat mui" (nasal congestion) before matching folded fainting terms.
    folded = f" {normalize_vietnamese(question)} ".replace(" ngat mui ", " ")
    return any(f" {phrase} " in folded for phrase in EMERGENCY_TERMS)


def is_personal_request(question):
    if not isinstance(question, str):
        return False
    text = f" {normalize_vietnamese(question)} "
    return any(f" {phrase} " in text for phrase in PERSONAL_TERMS)


is_missing_clinical_context = is_personal_request


def is_likely_out_of_domain(question):
    """Conservative domain hint used only after retrieval fails the dense gate.

    Returns True only when explicit out-of-domain cues match and no unambiguous
    health-domain terms are present; defaults conservatively to False
    (INSUFFICIENT_EVIDENCE) for ambiguous queries.
    """
    if not isinstance(question, str) or not question.strip():
        return False
    accented = f" {_normalize_accented(question)} "
    if any(f" {term} " in accented for term in ACCENTED_HEALTH_DOMAIN_TERMS):
        return False
    folded = f" {normalize_vietnamese(question)} "
    if any(f" {term} " in folded for term in HEALTH_DOMAIN_TERMS):
        return False
    return any(f" {term} " in folded for term in OUT_OF_DOMAIN_TERMS)


def strip_hidden_reasoning(text):
    """Remove hidden model thinking blocks (<think>...</think> and Gemma <unused94>thought...<unused95>)."""
    if not isinstance(text, str):
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _UNCLOSED_THINK_RE.sub("", cleaned)
    if "</think>" in cleaned.casefold():
        cleaned = _STRAY_CLOSE_THINK_RE.sub("", cleaned)
    cleaned = _GEMMA_THINK_BLOCK_RE.sub("", cleaned)
    cleaned = _GEMMA_UNCLOSED_THINK_RE.sub("", cleaned)
    if "<unused95>" in cleaned.casefold():
        cleaned = _GEMMA_STRAY_CLOSE_THINK_RE.sub("", cleaned)
    return cleaned.strip()


def contains_unsupported_medical_advice(text):
    """Detect explicit medication dosages or prescriptive personal advice in generated text."""
    if not isinstance(text, str) or not text.strip():
        return False
    if _DOSAGE_RE.search(text):
        return True
    normalized = f" {normalize_vietnamese(text)} "
    return any(f" {phrase} " in normalized for phrase in _PRESCRIPTIVE_PHRASES)


def validate_url(value):
    if (not isinstance(value, str) or not value
            or any(char.isspace() or ord(char) < 32 or char in '\\<>"\x60' for char in value)):
        return None
    try:
        url = urlsplit(value)
        if (url.scheme not in {"http", "https"} or not url.hostname
                or url.username is not None or url.password is not None):
            return None
        if url.port == 0:
            return None
        host = url.hostname.rstrip(".")
        labels = host.encode("idna").decode("ascii").split(".")
        if len(labels) < 2 or not all(re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?", label)
                                      for label in labels):
            return None
    except (ValueError, UnicodeError):
        return None
    return value
