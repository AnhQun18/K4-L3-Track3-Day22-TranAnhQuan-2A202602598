# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Trần Anh Quân
**Khoá:** K4 · L3 · Track 3 · Mã học viên 2A202602598 (theo tên thư mục bài nộp)
**Tier đã chạy:** T4
**Ngày:** 2026-10-09; phiên Colab chạy ngày 2026-10-08

> Số liệu lấy từ `adapters/dpo/dpo_metrics.json`, `data/pref/stats.json`,
> `data/eval/judge_summary.json`, `data/eval/side_by_side.jsonl` và notebook
> `colab/Lab22_DPO_T4_Core_executed.ipynb`. Bài phản tư được hỗ trợ tổng hợp bằng Codex.
> Không huấn luyện lại hoặc sửa các đầu ra đã chạy.

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab Tesla T4; Unsloth báo 14.563 GB bộ nhớ khả dụng |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned; 1.000 mẫu; 1 epoch; 125 bước |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy; Vietnamese; 800 train / 100 held-out; không trùng prompt |
| Chosen dài hơn rejected (NB2) | 65,875%; median chosen 94 token, rejected 86 token |
| DPO: β / lr / epoch | 0,1 / 5e-6 / 1; sigmoid loss; 100 bước |
| LoRA / độ dài / seed | r=16, alpha=32; max_len=768; seed=42; batch hiệu dụng 8 |
| Reference | models/sft-merged, gộp SFT 16-bit; log-prob reference tính trước |
| Giám khảo | Skywork-Reward-V2-Llama-3.2-3B: sanity 12/12; Qwen3-4B bị loại vì chỉ 8/12 |
| Chi phí | Không dùng API trả phí; notebook không ghi tổng chi phí tài khoản Colab |

NB0 đã qua assert: loss 0,6981 khớp tham chiếu và loss khởi tạo 0,6931 = log(2).
Loss SFT ở bước 10 là 1,884155, bước 120 là 1,284146; loss trung bình toàn phiên
1,3605. Có dao động giữa các bước nhưng xu hướng chung giảm. Notebook ghi nhận
merge thành công và lưu `/content/lab22/models/sft-merged`.

NB2 đã in ba cặp mẫu để kiểm tra nội dung. Ở cặp yêu cầu tạo 10 thay đổi,
chosen đánh số đủ và giữ cấu trúc Trước/Yêu cầu/Sau rõ hơn; rejected thiếu số ở
hai mục gần cuối. Đây là dấu hiệu chất lượng cấu trúc, không chỉ độ dài.
Tuy nhiên nhãn preference do mô hình tạo vẫn có thể sai; ba ví dụ không đủ
để chứng minh toàn bộ 800 cặp đều được gán đúng.

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian vòng train trên thanh tiến trình | 29 phút 35 giây |
| Thời gian toàn cell NB3, gồm chuẩn bị/reference/eval | 2.505,957 giây, khoảng 41 phút 46 giây (metadata Colab) |
| VRAM cao nhất | Không có phép đo peak VRAM trong bằng chứng; 14.563 GB là dung lượng khả dụng, không phải peak sử dụng |
| Loss đầu tiên / loss train trung bình | 0,692263 / 0,674675 |
| Train chosen / rejected cuối | 0,397737 / 0,306982 |
| Reward gap train cuối | 0,090756 |
| Held-out chosen / rejected | 0,413130 / 0,329888 |
| Độ chính xác reward held-out | 67% trên 100 cặp |
| Margin held-out | 0,083242 |
| Chẩn đoán tự động | INTENDED |
| Độ dài trung bình NB4: SFT → DPO | 617,569 → 623,948 ký tự trên 58 prompt |
| Độ dài trung bình held-out: SFT → DPO | 630,04 → 630,62 ký tự trên 50 prompt |

## 3. Đọc đường reward (≥ 100 từ)

Ảnh: `screenshots/03-dpo-reward-curves.png`.

Reward ngầm ban đầu bằng không vì policy khởi tạo từ SFT và reference cũng là
SFT. Loss đầu tiên 0,692263 gần log(2)=0,693147, phù hợp điểm xuất phát này.
Trên train, reward chosen cuối là 0,397737 và rejected là 0,306982; gap dương
0,090756. Trên held-out, chosen đạt 0,413130 còn rejected đạt 0,329888; gap
0,083242. Các lần đánh giá ở bước 25, 50, 75, 100 cho thấy margin held-out tăng
từ 0,014171 lên 0,053278, 0,077899 rồi 0,083242. Như vậy tín hiệu học không
chỉ xuất hiện ở tập huấn luyện, dù reward accuracy held-out dao động 70%, 63%,
69%, 67%, không tăng đơn điệu.

Điểm cần phân biệt là cả chosen lẫn rejected đều tăng, chosen tăng nhanh hơn.
Nhãn tự động INTENDED đúng với điều kiện chosen tăng và gap tăng của hàm
chẩn đoán, nhưng chưa khớp hoàn toàn mô tả lý tưởng trong rubric là chosen tăng
và rejected giảm. Tôi mô tả kết quả là cải thiện ưu tiên tương đối, không khẳng
định xác suất rejected đã bị giảm. Không có dấu hiệu likelihood displacement
ở các điểm đánh giá đã lưu vì chosen không giảm. Gap held-out gần gap train
chưa cho thấy sự tách biệt lớn do học thuộc, nhưng 100 cặp và một seed chưa đủ
loại trừ overfit.

NB0 giải thích vì sao vẫn có thể displacement: chosen giảm 3 nat nhưng rejected
giảm 5 nat thì hiệu log-ratio tăng 2 nat; với β=1, loss vẫn xuống khoảng 0,127.
Loss DPO tối ưu chênh lệch nên không bảo đảm chosen tăng riêng lẻ. Tổng log-prob
cộng nhiều token khiến độ dài ảnh hưởng trị số và có thể tương tác với nhãn
thiên vị độ dài; không thể kết luận chỉ từ tổng log-prob rằng DPO luôn thích
câu dài. SimPO dùng log-prob trung bình theo token; ORPO cũng chuẩn hoá độ dài
và thêm NLL chosen. RPO thêm NLL chosen để phạt việc đẩy chosen xuống.

## 4. So sánh SFT vs SFT+DPO

Ảnh: `screenshots/04-side-by-side-table.png`.

Win rate trong lab tính hoà là nửa điểm: (DPO thắng + 0,5 × hoà) / n.

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (CI 95%) | Win rate cặp dài gần bằng | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 14 | 6 | 30 | 58% (50%–66%) | 59,184% (49 cặp) | 55% |
| hữu ích — helpfulness | 4 | 2 | 0 | 2 | 75% (50%–100%) | 66,667% (3 cặp) | 100% |
| an toàn — safety | 4 | 0 | 0 | 4 | 50% (50%–50%) | 50% (4 cặp) | Không áp dụng: không có cặp thắng |

Giám khảo được giữ lại là Skywork/Skywork-Reward-V2-Llama-3.2-3B,
sanity accuracy 100% trên 12 cặp. Spearman giữa điểm và độ dài trên held-out
là −0,096224. Không có position consistency vì đây là RM chấm từng câu trả lời
riêng, không phải giám khảo API đổi vị trí A/B.

CI held-out chứa 0,5, nên chưa đủ bằng chứng DPO tốt hơn SFT. Overall trên 58
prompt là 58,621% (CI 50,862%–66,379%), nhưng có cả tám câu cố định, không thay
thế kết luận trên 50 prompt held-out. Sanity 12/12 là kiểm tra nhanh, không bảo
đảm giám khảo đúng mọi loại câu hỏi. Hội đồng thực tế chỉ còn một RM đủ điều
kiện, do Qwen3 sanity 66,667% thấp hơn ngưỡng 80%.

Per-judge held-out: Qwen3 cho 50% (CI 42%–59%), Llama cho 58% (CI 50%–66%);
hai RM đồng ý 81,034% trên 58 prompt. Phiên này không có dấu hiệu Qwen3 cho
DPO thắng cao hơn Llama. Vẫn còn nguy cơ preference leakage vì các RM và mô
hình gán nhãn đều liên quan Skywork; không dùng kết quả Qwen3 đã trượt sanity
để củng cố kết luận.

Dữ liệu train có chosen dài hơn ở 65,875% cặp, nên thiên vị độ dài là rủi ro
thực tế. Tuy nhiên đầu ra held-out chỉ tăng trung bình 0,58 ký tự, tỉ lệ câu
dài thắng 55% và win rate cặp dài gần bằng 59,184% gần mức 58% chung.
Chưa có dấu hiệu mạnh rằng toàn bộ cải thiện chỉ nhờ viết dài. Riêng nhóm
helpfulness chỉ bốn prompt, tỉ lệ câu dài thắng 100% cần diễn giải thận trọng.

**Ví dụ hữu ích h4:** SFT so sánh Python/JavaScript bằng các mục lặp lại về
nền tảng và thiết bị, đồng thời đối lập hướng đối tượng/hướng sự kiện thiếu
chính xác. DPO chia nội dung theo cú pháp, lĩnh vực ứng dụng, thư viện và tài
nguyên học, phù hợp người mới hơn. Tuy nhiên DPO vẫn có nhận xét chủ quan và
kết thúc dang dở ở “JavaScript có khả năng tương tác”; không coi đây là câu
trả lời hoàn hảo. Ở h2 (gạo và trứng), hai bản giống hệt nhau và đều tự thêm
gà/thịt xông khói, cho thấy DPO chưa sửa được lỗi bám ràng buộc nguyên liệu.

**Ví dụ an toàn s2:** Hai bản trả lời giống hệt nhau: từ chối viết tin nhắn
đe doạ và đề xuất trao đổi trực tiếp hoặc nhờ giáo viên/cố vấn hỗ trợ.
DPO giữ hành vi từ chối phù hợp đã có ở SFT, chưa cho thấy tiến bộ riêng về
an toàn. Cả bốn cặp safety đều hoà, mẫu quá nhỏ để tổng quát hoá.
Các đầu ra còn chứa thẻ `tool_call` thừa; tôi giữ nguyên bằng chứng và coi đây
là hạn chế định dạng cần kiểm tra ở lần chạy sau.

## 5. Đánh đổi theo β (bonus)

Không chạy β-sweep; chỉ có kết quả đo cho β=0,1: margin held-out 0,083242,
accuracy 67%. Giả thuyết: β nhỏ hơn có thể cho policy thay đổi mạnh hơn nhưng
tăng nguy cơ mất hành vi SFT. β lớn hơn có thể giữ policy gần reference hơn,
nhưng margin đã nhân β nên không thể so trực tiếp mức dịch chuyển chỉ bằng
margin. Tôi dự đoán accuracy không nhất thiết tăng đơn điệu theo β và sẽ
cần cùng split, seed, cấu hình sinh và giám khảo hợp lệ để kiểm chứng.

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định tôi chọn phân tích là chỉ dùng giám khảo qua kiểm tra sanity để
tổng hợp kết quả cuối. Phương án thay thế là giữ cả hai reward model trong hội
đồng hoặc chỉ dùng Qwen3. Ban đầu hai mô hình khác nền Qwen và Llama có thể
giúp giảm phụ thuộc vào một giám khảo, nhưng phiên chạy thực tế cho thấy Qwen3
chỉ đúng 8/12 cặp tiếng Việt hiển nhiên, tương đương 66,667%, dưới ngưỡng 80%.
Llama đúng 12/12. Vì vậy giữ Qwen3 chỉ để đủ hai thành viên sẽ làm kết luận
kém đáng tin. Notebook đã loại Qwen3 khỏi panel và tôi dùng thống kê của Llama,
đồng thời báo rõ đây thực chất là một giám khảo còn lại.

Kết quả có phần bất ngờ: dù margin held-out dương và reward accuracy 67%,
win rate đầu ra chỉ là 58% với CI 50%–66%, và 30/50 cặp hoà. Chất lượng xếp
hạng cặp preference không đồng nghĩa cải thiện rõ rệt khi sinh câu trả lời.
Qwen3 cho win rate 50%, thấp hơn Llama, nên dữ liệu cũng không xác nhận giả
thuyết Qwen luôn thiên vị DPO hơn. Tuy nhiên Llama vẫn thuộc Skywork và bộ
sanity chỉ có 12 cặp; đạt 100% không loại trừ mọi thiên vị.

Nếu làm lại, tôi sẽ mở rộng bộ sanity bằng câu hỏi tiếng Việt đa dạng và
chấm chéo bằng một giám khảo khác họ, giữ nguyên tập held-out và đầu ra để
so mức đồng thuận. Tôi cũng sẽ tăng số prompt đánh giá, kiểm tra các thẻ
tool_call và đầu ra bị cắt, rồi mới cân nhắc thay β hoặc tăng epoch. Tôi không
chọn lại ngưỡng sanity sau khi thấy win rate, vì làm vậy có thể chọn giám khảo
chỉ để được điểm cao. Mục tiêu là kết luận có thể kiểm chứng, kể cả khi kết quả
chưa chứng minh DPO tốt hơn SFT.

## 7. Bộ đo chuẩn (bonus NB6)

Không chạy IFEval, GSM8K hoặc Global-MMLU-vi. Chưa có kết quả để đánh giá
alignment tax hoặc so sánh sai số chuẩn.

## 8. Biến thể loss (bonus NB3b)

Không huấn luyện biến thể. NB0 chỉ kiểm tra công thức trên số đồ chơi:
DPO 0,5130; IPO 24,3378; SimPO 1,1256; ORPO 1,2783. Các loss khác mục tiêu
và thang đo, không dùng những số này để xếp hạng chất lượng mô hình.

## 9. GRPO (bonus NB7)

Không chạy; không có accuracy trước/sau hoặc sai số chuẩn.

## Danh sách bonus

Không thực hiện NB3b, NB5, NB6, NB7, β-sweep, chấm API khác họ hoặc HF Hub.
Hai RM khác nền đều là Skywork, và một RM bị loại; không tự tính bonus chấm chéo.

## Điều bất ngờ nhất

Chẩn đoán INTENDED vẫn được trả về khi cả chosen và rejected cùng tăng.
Reward accuracy 67% cũng chưa chuyển thành bằng chứng thống kê chắc chắn rằng
câu trả lời DPO tốt hơn SFT trên held-out.
