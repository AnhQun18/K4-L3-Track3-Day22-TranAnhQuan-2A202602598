# Bằng chứng bài nộp Lab 22

Nguồn: [notebook Colab đã chạy](https://colab.research.google.com/drive/1XP8wNRxWtdxJ84-OZL8P5l3eGsZE2Cb5?usp=sharing).

- Notebook gốc được lưu nguyên output tại `colab/Lab22_DPO_T4_Core_executed.ipynb` (105 cell).
- JSON, JSONL, Parquet và bốn ảnh được giải nén từ `lab22-evidence.zip` của phiên chạy. Ảnh bảng NB4 được render lại từ JSONL bằng `python scripts/render_comparison.py` để không cắt chữ; không thay nội dung câu trả lời.
- `REFLECTION.md` đã điền sau phiên Colab. Dòng verify thất bại trong notebook gốc là kết quả lịch sử trước khi điền phản tư; không sửa thành output giả.
- `notebooks/00_dpo_loss_from_scratch.py` đã điền loss và assert log(2), tương ứng lời giải đã chạy trong Colab.
- Không có trọng số SFT/DPO trong repo. Notebook có output merge SFT thành công; cấu hình adapter vẫn giữ đường dẫn Colab gốc.
- Verifier chấp nhận bản Colab xuất ra khi có output merge thành công đúng đường dẫn và metrics NB3 trùng JSON hiện tại; vẫn kiểm tra fingerprint split và SHA-256 đầu ra NB4.
- Chạy kiểm tra từ gốc repo: `python scripts/verify.py`.
- Kiểm tra verifier: `python -m unittest discover -s scripts -p test_submission.py`.
- Bản Colab gốc vẫn ghi REFLECTION mẫu trong cell setup. Khi chạy lại, thay nội dung cell đó bằng phản tư cuối hoặc điền lại sau khi có số liệu phiên mới; không dùng số liệu phiên cũ cho phiên mới.

Kết luận: DPO reward accuracy 67%; win rate held-out 58%, CI 95% [50%, 66%].
Chưa đủ bằng chứng DPO tốt hơn SFT. Qwen3 RM bị loại do sanity 8/12;
Llama RM được giữ lại với 12/12. Bonus chưa chạy.

Thông tin học viên trong phản tư suy ra từ tên repo/thư mục: Trần Anh Quân, K4, mã 2A202602598.

Kiểm tra cuối tại máy nộp: verifier đạt, 5 kiểm thử portability đạt. Python máy này
thiếu PyTorch và engine Parquet, nên không chạy lại NB0 hoặc đọc Parquet độc lập;
assert loss và không trùng prompt đã qua trong output Colab. Verifier đối chiếu
fingerprint của hai file Parquet và SHA-256 đầu ra NB4. Không chạy lại GPU pipeline.
