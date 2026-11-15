# Tình hình dự án — 15/11/2026

Tóm tắt để tiếp tục trên máy khác. Chi tiết: `docs/CG_SPLIT.md` (protocol + mọi kết quả CG), `docs/METHOD.md` và
`docs/METHOD_RATIONALE.md` (HGO v3), `docs/RESEARCH_DESIGN.md` (thiết kế thí nghiệm).

## 1. Câu hỏi nghiên cứu

Vấn đề: **compositional generalization (CG)** của planner end-to-end từ camera. Model đã thấy từng thành phần (ví dụ "rẽ ở
giao lộ" và "người đi bộ cắt qua") nhưng chưa thấy chúng cùng lúc thì có lái kém đi không. Method đề xuất: HGO v3 (biểu diễn
lai register toàn cục + slot theo tác nhân, world model có điều kiện hành động, scorer theo tác nhân). Brief ban đầu đặt
bằng chứng CG chính ở **CARLA closed-loop**; NAVSIM chỉ là kiểm tra planning trên dữ liệu thật.

Đại lượng chính là **exposure gap** = điểm(model train có tổ hợp) − điểm(cùng model, cùng ngân sách, train không có tổ hợp),
trên cùng cảnh test của tổ hợp. So "cảnh lạ vs cảnh thường" (kiểu Fail2Drive) lẫn độ khó, không đủ.

## 2. Đã làm trên NAVSIM

**Split CG v1** (`docs/CG_SPLIT.md` §1–8): tự gán nhãn mọi frame từ GT + bản đồ (hành động trong 10 s × tác nhân cắt qua
đường đi, PET < 3 s, xe tới điểm xung đột trong 6 s); chia theo drive session (137 train / 34 test phụ; navtest 46 session
khác); pool train 84.390 cửa sổ; fold chính **rẽ ở giao lộ × người đi bộ / xe đạp cắt qua**: withheld bỏ 2.600 cửa sổ
(purge rộng), exposed bỏ cùng số cửa sổ không đối tác; sau lọc cache mỗi bản 81.078. Test: 241 cảnh của ô (32 navtest +
209 test phụ, 50 log) + nhóm đối chứng (same_man, same_partner, man_none, other), tổng 2.621 cảnh.

**Bước 0** (§9) — DrivoR công bố (đã thấy tổ hợp) trên các nhóm test: ô kém hơn cảnh thường (EPDMS 85,4 vs 91,0; va chạm có
lỗi ~7,7 % vs ~0,7 %). EPDMS thấp do **rẽ** (tiến độ, giữ làn; rẽ không ai cắt qua cũng 85,6); va chạm do **người cắt qua**.
Chưa chứng minh tổ hợp tệ hơn cộng dồn hai thành phần.

**Bước 1** (§10–10.1) — cặp DrivoR withheld / exposed, 812×476, 6 epoch, 30.408 bước mỗi run, 1 seed, ~37,6 GPU-giờ
(~2,9 triệu đồng): gap trên ô **+0,31 EPDMS [−1,76; 2,56]**, NC −1,24 [−3,19; 0,48]; mọi hiệu của hiệu chứa 0. Bản exposed
ngang bản công bố (pilot train bình thường).

**Kiểm tra liều** (§10.2, chỉ CPU) — chấm hai checkpoint trên cửa sổ **train**: exposed không tốt hơn withheld ngay trên 746
cửa sổ ô nó đã được train (−0,15 EPDMS, NC 0,00); withheld cũng không tốt hơn trên cửa sổ riêng của nó. Theo quy tắc chốt
trước: **phép đo không nhạy**. Kết luận: "gap ≈ 0" ở bước 1 **không** phải bằng chứng không có gap CG; thí nghiệm chưa trả
lời được câu hỏi.

Nguyên nhân có thể (chưa tách được):
1. Liều nhỏ: 746 cửa sổ = 0,9 % dữ liệu.
2. DrivoR chọn quỹ đạo bằng scorer học nhãn PDM của mọi quỹ đạo thử trong mọi cảnh → học quy tắc chung "đừng cắt qua chỗ
   người sắp đi tới", không cần ví dụ expert của tổ hợp. Bản withheld hiện chỉ bỏ ví dụ expert (**B-L1**), chưa che nhãn
   của các quỹ đạo thử rơi vào ô (**L2**; cả hai = **B-L12**).
3. NAVSIM open-loop 4 s, tác nhân không phản ứng: phần chờ / nhích / rẽ khi người đã qua nằm ngoài cái được chấm; tín hiệu
   chỉ đến từ ~8–11 cảnh va chạm.

Giới hạn khác đã biết: 1 seed (CI bootstrap chỉ đo bất định tập test); purge rộng còn bỏ một phần thành phần (VRU cắt qua khi
không rẽ: 415 ở withheld vs 481 ở exposed — làm withheld kém đi, không giải thích được gap ≈ 0); tách session không bảo đảm
giao lộ test là địa điểm mới.

## 3. Việc nên làm tiếp: kiểm định protocol trước khi train thêm

Nguyên tắc: chỉ train model khi bộ test + metric + thống kê đã được chứng minh **bắt được** gap khi gap có thật. Các bước
rẻ, không cần GPU (áp dụng cho NAVSIM, và lặp lại cho CARLA):

1. **Metric có phạt đúng hành vi của tổ hợp không** (không train): chấm các planner cố định trên 2.621 cảnh — quỹ đạo người
   lái thật, PDM-Closed (luật, thấy mọi tác nhân), PDM-Closed **không thấy người đi bộ / xe đạp**, vận tốc không đổi. Nếu
   planner "mù VRU" không tụt rõ trên ô và same_partner (so với man_none / other) thì EPDMS 4 s không đo được việc nhường
   người trong các cảnh này → bộ test không dùng được cho CG. (Cần kiểm tra cách che VRU khỏi input của PDM-Closed.)
2. **Đối chứng dương cho exposure** (gần như không train): một planner "học thuộc", ví dụ tra cứu láng giềng gần nhất (kNN)
   trên đặc trưng cảnh đơn giản, dựng từ pool withheld và pool exposed. Planner này chỉ làm được những gì có trong pool, nên
   phải có gap trên ô. Không thấy gap → bộ test / metric không phát hiện được việc thiếu ví dụ → đừng train model đắt trên nó.
3. **Cỡ hiệu ứng tối thiểu** (power): với 241 cảnh / 50 log, phát hiện được gap cỡ ≥ 3 điểm EPDMS; chốt trước gap bao nhiêu
   là có ý nghĩa.

Chỉ khi 1 và 2 qua: chạy DrivoR B-L12 (1 run withheld mới, ~19 giờ GPU, ~1,5 triệu đồng; cần code che nhãn quỹ đạo thử
trong loss DrivoR, dùng lại `src/ade2e_wm/hgo/holdout.py`), rồi mới thêm seed / method / fold.

## 4. CARLA

- Fail2Drive (`ade2e-paper/Fail2Drive_2604.08535.pdf`) có bộ công cụ tạo cảnh và expert PDMLite-F2D để thu dữ liệu: dùng
  được để dựng split "train thiếu tổ hợp / test closed-loop trên tổ hợp". Gap của Fail2Drive là "cảnh lạ vs cảnh thường"
  trên checkpoint có sẵn, không phải exposure gap.
- DrivoR không train trên CARLA. Method train được trên CARLA để đo exposure gap: **PlanT / PlanT 2.0** (đầu vào là đối
  tượng, train rẻ — hợp làm đối chứng dương và kiểm định protocol nhanh), **TransFuser++** (camera + LiDAR, carla_garage),
  TCP / UniAD / VAD bản Bench2Drive. Exposure gap cần train lại mỗi method hai lần (có / không có tổ hợp).
- Ghi chú khả thi CARLA cũ đã xoá; cần kiểm tra lại trên máy mới.

## 5. Code và dữ liệu

| Việc | File |
|---|---|
| Nhãn tình huống, split, manifest | `scripts/hgo/cg_events.py`, `cg_cells.py`, `cg_split.py` |
| Chuẩn bị cặp DrivoR (lọc cache, cân bằng, tập chấm) | `scripts/hgo/cg_drivor_prep.py` |
| Train DrivoR trên fold (resume có kiểm danh tính) | `scripts/hgo/cg_drivor_train.sbatch`, `cg_drivor_run_training.py` |
| Chấm EPDMS trên tập test / cửa sổ train | `scripts/hgo/cg_drivor_eval.sbatch`, `cg_dose_eval.sbatch` |
| Gap + bootstrap + hiệu của hiệu; kiểm tra liều | `scripts/hgo/cg_gap.py`, `cg_dose.py` |
| HGO v3 | `src/ade2e_wm/hgo/`, `scripts/hgo/train.py` (còn đọc manifest chưa lọc cache; purge chưa phủ frame lịch sử) |

Không có trong git (cần copy nếu dùng lại): `data/_lists/cg/` (split_v1 15 MB + events parquet 105 MB; bỏ qua `old_v0`),
`exp/cg_eval/` (CSV điểm và json gap, 7,4 MB), checkpoint pilot `exp/cg_drivor/*/lightning_logs/*/checkpoints/last.ckpt`
(300 MB mỗi cái), metric cache `exp/metric_cache/` (lớn). Bốn repo `third_party/` clone lại theo commit ghi trong README.

Cluster cũ: chỉ chạy qua sbatch, không chạy gì trên login node; ≤ 2 job; partition main tính ×2, tháng 11 đã ~13,5 triệu
đồng; ~2,2 giây/bước DrivoR, 84 % là chấm PDM trên CPU.
