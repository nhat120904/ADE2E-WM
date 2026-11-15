# Split CG trên NAVSIM (Protocol B) — split v1, 12/11/2026

Mục tiêu: đo **mức tụt do chưa thấy tổ hợp** (Δ_exposure) cho mọi planner, train lại trên cùng dữ liệu, không rò, đủ cảnh test.
Dùng trong RESEARCH_DESIGN.md §3 (Protocol B) và Bảng 2. Bản nháp cũ (split pilot `turn_vru`, định nghĩa lỏng):
`exp/archive/docs_2026-11-12/`.

## 1. Nhãn tình huống

Nhãn chỉ từ annotation GT và bản đồ, không bao giờ là input của model; tính cho **mọi frame** của 1.310 log trainval và 147 log
test bởi `scripts/hgo/cg_events.py` → `data/_lists/cg/events_{trainval,test}.parquet`.

- **Hành động** (đường đi đã log của xe mình, tương lai liên tục ≤ 10 s): dừng (< 2 m), thẳng, trái / phải (đổi hướng > 20°).
  Trong ô, rẽ trái / phải phải đi qua lớp **giao lộ** của bản đồ nuPlan; đổi hướng ngoài giao lộ là `curve` (đường cong).
- **Đối tác**: tác nhân có quỹ đạo **cắt qua** đường đi của xe (có vị trí ở cả hai bên đường đi, trong 8 m), với PET < 3 s (hai
  bên đi qua cùng điểm cách nhau dưới 3 s) và xe tới điểm xung đột trong 6 s. Nhóm: `vru` (người đi bộ, xe đạp), `veh_opp` (xe
  hướng ngược lại tại điểm xung đột, > 135°), `veh_cross` (45–135°); `veh_same` báo riêng.
- **Bối cảnh** (báo cáo, không dùng để giữ lại): thành phố, mật độ, tốc độ, lệnh điều hướng, giao lộ.

Định nghĩa "cắt qua" và "rẽ trong giao lộ" được chọn sau khi xem ảnh sample: định nghĩa "tác nhân đi gần đường đi" đếm nhầm xe
ngược chiều ở làn bên cạnh, người đi dọc mép đường hoặc đi sau xe, và đường cong không phải giao lộ.

**Ô** = (hành động × đối tác). Một fold giữ lại một ô (hoặc một nhóm ô); mọi ô khác vẫn train. Điều kiện support (SVIB,
Wiedemer et al.): sau khi purge, cả hai thành phần của ô vẫn có trong train (cùng hành động với đối tác khác; cùng đối tác với
hành động khác).

## 2. Giữ lại ở ba mức

| Mức | Cái gì bị bỏ | Cách |
|---|---|---|
| L1 — hành vi expert + target | cửa sổ train có t0 **hoặc** bất kỳ frame nào trong 8 frame tới thuộc ô, theo định nghĩa **rộng** (hành động 4 s hoặc 10 s, không xét giao lộ; mọi tác nhân đi gần đường đi, cắt qua hay không) | lọc danh sách token trước khi train, mọi method |
| L2 — nhãn ứng viên | trong cửa sổ còn lại, ứng viên có hành động của ô và xung đột với đối tác của ô (theo tương lai GT 4 s) | mask trong loss scorer, mọi readout và mọi baseline |
| L3 — tiền huấn luyện | DINOv2 chung, không kiểm soát được | ghi rõ: "chưa thấy trong huấn luyện lái xe" |

Định nghĩa rộng của purge là tập cha của ô chặt, nên purge thận trọng. Hai biến thể: **B-L1** và **B-L12**. Nếu ứng viên phản
thực tạo support cho tổ hợp thì ở B-L1 mọi model tụt ít, ở B-L12 model nguyên khối tụt nhiều hơn readout có cấu trúc.

## 3. Mốc "được thấy" (exposed)

Cùng pool, cùng số cửa sổ, cùng số bước train: thay vì bỏ cửa sổ của ô, bỏ cùng số cửa sổ ngẫu nhiên **không có đối tác** (giữ
nguyên các thành phần); không mask L2. Δ_exposure = S(exposed, cảnh test của ô) − S(withheld, cùng cảnh). Để tiết kiệm GPU có thể
dùng model train trên **toàn pool** làm exposed chung (lệch ngân sách 3,1 % với fold chính) và làm luôn model của Bảng 1.

## 4. Chia theo drive session và pool train

- 171 session trainval → **34 session test CG phụ** (20 %, phân tầng theo thành phố, seed 0: Singapore 3, Boston 7, Las Vegas
  18, Pittsburgh 6; 275 log) và **137 session train** (1.035 log). navtest gồm 46 session khác. Không session nào vừa train vừa
  test, nên cảnh test không chung đoạn đường, thời điểm, người tham gia giao thông với train.
- **Pool train chung cho mọi method:** 84.390 cửa sổ từ 988 log = 79.111 token navtrain của session train + 5.279 frame có xung
  đột ngoài navtrain (thưa 1 s) để giữ support. 23.283 token navtrain nằm trong session test bị bỏ (pool ≈ 77 % navtrain). Mọi
  cửa sổ có 4 frame lịch sử và 10 frame tương lai (quy tắc của devkit).

## 5. Tập test và cỡ mẫu

- Cảnh test = cảnh navtest + 5.071 frame của session test phụ (mọi ô có đối tác, thưa 1 s; thêm 1.500 frame không đối tác cho
  mỗi hành động làm đối chứng). Chấm bằng evaluator chính thức (cần metric cache, §9).
- Mỗi fold chia cảnh test thành nhóm: `cell` (ô giữ lại), `same_man` (cùng hành động, đối tác khác), `same_partner` (cùng đối
  tác, hành động khác), `man_none` (cùng hành động, không đối tác), `other`. Nhóm cha dùng để tách độ khó khỏi exposure.
- Cỡ mẫu: MDE80 = 2,8 × SE của hiệu hai model ghép cặp (bootstrap theo log). Hiện chỉ ước được từ phần navtest của fold (H32 −
  DrivoR), nên còn thô; đo lại khi có điểm thật trên cảnh test phụ. Tiêu chí: MDE80 ≤ 4 điểm EPDMS, ≥ 30 log test.

## 6. Tiêu chí chọn fold

Theo thứ tự: (1) cảnh test đủ (§5); (2) support của hai thành phần sau purge ≥ 1.000 cửa sổ từ ≥ 50 log; (3) purge ≤ 10 % pool;
(4) đa dạng: một fold VRU, một fold xe; (5) một fold mà cấu hình tương đối (vị trí, hướng của đối tác tại điểm xung đột) không
được ô khác phủ, nơi readout ghép cũng được dự đoán là hỏng. Chốt định nghĩa và fold **trước** khi nhìn kết quả model nào trên
chúng.

## 7. Fold v1

| Fold | Cảnh test (navtest + phụ) | Log / session test | Purge | Support đối tác khi không làm hành động của ô (cửa sổ / log) | MDE80 ước |
|---|---|---|---:|---|---:|
| **rẽ trái / phải ở giao lộ × người đi bộ / xe đạp cắt qua (chính)** | 32 + 209 = 241 | 50 / 28 | 3,1 % | 432 / 85 | ~3,8 |
| rẽ trái ở giao lộ × xe ngược chiều (tham khảo) | 26 + 71 = 97 | 26 / 20 | 1,1 % | 225 / 44 | ~3,3 |
| thẳng × VRU (loại) | 62 + 131 = 193 | 46 / 30 | 5,4 % | 274 / 80 | |

- Fold chính gộp rẽ trái và rẽ phải (user chọn 12/11) để có đủ cảnh: tách riêng thì rẽ trái 101 cảnh, rẽ phải 140 cảnh.
- **Tổ hợp thật hiếm trong log thực:** nếu chỉ dùng navtest, fold chính có 32 cảnh. Đây là lý do thêm session test phụ.
- Điểm yếu: VRU cắt qua khi xe không rẽ chỉ có 432 cửa sổ chặt (dưới tiêu chí 1.000). Pool vẫn còn nhiều cửa sổ có VRU gần đường
  đi nhưng không cắt qua, nên thành phần "người đi bộ" được thấy rộng hơn con số này.
- Fold xe chỉ tham khảo vì support 225 cửa sổ.
- Loại thẳng × VRU: purge rộng (hành động 4 s = thẳng) xoá phần lớn cửa sổ rẽ có VRU.

## 8. Kiểm tra

- Không token nào của session test trong pool; không token chung giữa pool, test phụ và navtest.
- Bản withheld không còn cửa sổ nào của ô (cả định nghĩa chặt lẫn rộng); exposed và withheld cùng số cửa sổ; exposed giữ đủ
  cửa sổ của ô.
- Chạy lại cho kết quả giống hệt (seed cố định).
- Ảnh sample (3 camera + bản đồ nhìn từ trên vẽ từ nhãn GT): `docs/figures/cg_samples/`.

## 9. Bước 0: DrivoR công bố trên các nhóm test (13/11)

Checkpoint DrivoR Nav2 công bố (train trên toàn navtrain, **đã thấy** tổ hợp; đã thấy một phần các session test phụ, nên số
lạc quan), EPDMS NAVSIM v2.2 one-stage chính thức trên navtest + 5.030 / 5.071 frame test phụ có metric cache. Đây là đo
**độ khó / chỗ còn cải thiện**, không phải gap do chưa thấy. Script `scripts/hgo/cg_eval_drivor.sbatch`,
`scripts/hgo/cg_group_scores.py`; kết quả `exp/cg_eval/drivor_pub_split_v1/groups_<fold>.json`.

Fold chính (rẽ ở giao lộ × người đi bộ / xe đạp cắt qua), navtest + test phụ:

| Nhóm | Cảnh / log | EPDMS [CI 95 %] | NC (không va chạm có lỗi) | EP |
|---|---|---|---:|---:|
| cell | 241 / 50 | 85,4 [81,1; 89,3] | 92,3 | 88,3 |
| rẽ, đối tác khác | 180 / 38 | 88,2 [83,6; 92,0] | 93,9 | 89,0 |
| người đi bộ cắt qua, không rẽ | 200 / 50 | 86,6 [82,0; 90,4] | 95,5 | 79,6 |
| rẽ, không đối tác | 6.760 / 317 | 85,6 [84,6; 86,8] | 98,6 | 78,9 |
| cảnh khác | 9.795 / 379 | 91,0 [90,4; 91,6] | 99,3 | 84,5 |

Hiệu cell − nhóm (bootstrap theo log): so với cảnh khác EPDMS −5,6 [−10,2; −1,3], NC −6,9 [−11,6; −2,7]; so với rẽ không đối
tác EPDMS −0,2 [−5,1; 4,0] nhưng NC −6,3 [−10,5; −2,2]; so với từng thành phần riêng (rẽ × đối tác khác, người đi bộ không
rẽ) NC −1,6 và −3,2, chưa có ý nghĩa thống kê. Fold xe ngược chiều: NC cell 91,7 so với 98,9 ở cảnh khác.

Đọc: (1) DrivoR dù đã thấy tổ hợp vẫn va chạm có lỗi ở ~7,7 % cảnh của ô (≈ 18 / 241), so với 1,4 % khi rẽ không có ai cắt
qua: chỗ còn cải thiện nằm ở **an toàn**, đúng phần scorer theo tác nhân nhắm tới. (2) EPDMS của ô thấp chủ yếu do rẽ (cảnh
rẽ nào cũng thấp vì EP / LK); riêng tổ hợp không làm EPDMS tụt thêm. (3) Chưa chứng minh được ô tệ hơn **từng thành phần**
của nó (tác động siêu cộng): CI rộng với 241 cảnh. Phần navtest sạch chỉ có 32 cảnh, mọi hiệu đều không có ý nghĩa. Gap do
chưa thấy cần bước 1 (cặp withheld / exposed).

## 10. Bước 1: cặp DrivoR withheld / exposed (pilot, giao thức chốt 14/11 trước khi có kết quả)

Mục đích: đo xem một method hiện có có tụt ở ô khi chưa thấy tổ hợp hay không (go / no-go cho HGO). Đã qua một review
trước khi chạy (14/11, file review đã xoá 15/11).

- **Dữ liệu train**: manifest fold turn_vru sau lọc metric cache, **cân bằng** cùng số cửa sổ (bỏ ngẫu nhiên, seed 0, phần
  dư trong các cửa sổ đối chứng chỉ có ở withheld; `balance_dropped.txt`), lưu thành `train_<variant>_cached.txt` dùng
  chung cho mọi method. Job train kiểm tra số mẫu loader dựng ra đúng bằng N.
- **Mức giữ lại: B-L1** (cửa sổ expert). Scorer của DrivoR vẫn học nhãn PDM của các quỹ đạo phản thực, trong đó có thể có
  tổ hợp giữ lại. Chiều ảnh hưởng lên gap **không được bảo đảm** (giám sát phản thực có thể giúp withheld, nhưng phân bố
  proposal và quá trình tối ưu cũng đổi). Không gọi là "chưa thấy tổ hợp ở mọi giám sát", không so ngang với HGO B-L12.
- **Recipe pilot**: code và loss DrivoR gốc, 812 × 476, 6 epoch, batch 16, 1 GPU, seed 0, nhãn train PDM v2.2 qua
  `ade2e_wm.hgo.pdm` (giống HGO); checkpoint cuối, không chọn checkpoint hay siêu tham số bằng tập test. Ngân sách cố định 6 × ceil(81.078 / 16) = 30.408 bước optimizer (khóa bằng max_steps, kiểm tra khi kết thúc); resume giữa epoch không khôi phục đúng thứ tự mẫu.
- **Chấm**: EPDMS v2.2 one-stage chính thức trên tập rút gọn cố định (`eval_subset.txt`: mọi cảnh có metric cache của cell /
  rẽ đối tác khác / người đi bộ không rẽ + 1.000 cảnh ngẫu nhiên mỗi nhóm đối chứng lớn, trộn navtest và test phụ). Mọi
  model chấm trên đúng tập này. Thiếu cảnh nào thì `cg_gap.py` dừng.
- **Nhóm báo cáo** (chốt trước): cell; **cell_in4** = cùng quy tắc strict của ô nhưng trên tương lai 4 s (`man4` rẽ có giao
  lộ `inter4`, tác nhân đầu tiên cắt qua trong 4 s là người đi bộ / xe đạp, `petx4` < 3 s); **cell_not_strict4** = phần còn lại
  của cell (chưa chứng minh tất cả là chuẩn bị tương tác: 79 cảnh có rẽ trong 4 s và xung đột của nhãn 10 s trong 4 s nhưng
  không qua phép cắt qua 4 s). Nhãn ô nhìn tới 10 s còn EPDMS chấm 4 s, nên hai nhóm con
  được báo riêng. Kiểm tra 14/11 (job 65456): cell_in4 chỉ có **15 / 241** cảnh (10 session), cell_not_strict4 226. Quét rộng
  hơn trong 34 session test phụ + navtest chỉ được 28 cảnh / 12 log. cell_in4 chỉ để mô tả; pilot đo exposure gap cho ô
  "tình huống nhìn tới 10 s, hành vi chấm trong 4 s", không chứng minh việc thực hiện trọn tổ hợp trong 4 s.
- **Thống kê**: gap = exposed − withheld theo nhóm (EPDMS, NC), bootstrap cặp theo log và theo session; hiệu của hiệu
  gap(cell / cell_in4 / cell_not_strict4) − gap(nhóm đối chứng). Lần bootstrap không có cảnh của nhóm là không xác
  định (bỏ, báo số lần hợp lệ), không tính là gap 0.
- **Chất lượng exposed**: chấm thêm checkpoint DrivoR công bố trên cùng tập rút gọn (cùng ngữ cảnh chấm, vì EPDMS có thành
  phần comfort giữa các frame) để thấy pilot kém bao nhiêu so với bản đầy đủ.
- **Giới hạn kết luận**: 1 seed, 1 baseline, recipe giảm, val trong lúc train là cảnh train (chỉ theo dõi). Kết quả chỉ đủ
  cho quyết định go / no-go, chưa đủ để kết luận chung về các method hiện nay: kết quả chính cần thêm seed, thêm baseline.

### 10.1 Kết quả pilot (15/11, seed 0)

Train: job 65462 (withheld) / 65463 (exposed), mỗi run 30.408 bước (đúng ngân sách), ~18 giờ 50 phút, ~37,6 GPU-giờ
cả cặp. Chấm: 65499 / 65500 (CPU), checkpoint công bố 65502. Đủ 2.621 / 2.621 cảnh cho cả hai run (không thiếu cảnh nào).
Gap = exposed − withheld, CI 95 % bootstrap theo log (theo session cùng kết luận), 2.000 lần, 2.000 lần hợp lệ.
File: `exp/cg_eval/gap_drivor_turn_vru_s0.json`, `exp/cg_eval/drivor_turn_vru_{withheld,exposed}_812x476_e6_s0/`,
`exp/cg_eval/drivor_pub_subset_turn_vru/`.

| nhóm | n | EPDMS ex / wh / công bố | gap EPDMS | gap NC |
|---|---|---|---|---|
| cell | 241 | 87,11 / 86,80 / 85,47 | +0,31 [−1,76; 2,56] | −1,24 [−3,19; 0,48] |
| cell_in4 (mô tả) | 15 | 88,12 / 87,98 / – | +0,15 [−0,58; 0,75] | 0,00 |
| cell_not_strict4 | 226 | 87,05 / 86,72 / – | +0,32 [−1,90; 2,74] | −1,33 [−3,41; 0,50] |
| same_man | 180 | 90,52 / 89,81 / 88,24 | +0,72 [−1,54; 3,13] | +0,56 [−1,14; 2,76] |
| same_partner | 200 | 85,18 / 87,36 / 86,60 | −2,18 [−4,33; −0,52] | −1,50 [−3,68; 0,00] |
| man_none | 1.000 | 85,94 / 86,37 / 86,93 | −0,43 [−1,33; 0,52] | −0,25 [−0,93; 0,39] |
| other | 1.000 | 90,07 / 90,34 / 90,68 | −0,27 [−1,13; 0,53] | +0,20 [−0,32; 0,74] |

Hiệu của hiệu (gap cell − gap đối chứng), EPDMS / NC: same_man −0,41 [−3,82; 3,16] / −1,80 [−4,51; 0,60];
same_partner +2,49 [−0,42; 5,62] / +0,26 [−2,29; 2,93]; man_none +0,74 [−1,51; 3,19] / −0,99 [−2,97; 0,91];
other +0,58 [−1,50; 2,85] / −1,44 [−3,45; 0,38].

Đọc: không thấy exposure gap trên ô turn_vru với DrivoR (1 seed); mọi hiệu của hiệu chứa 0. same_partner có gap âm
có ý nghĩa (−2,18) không có cơ chế giải thích rõ; có thể là dao động giữa hai lần train, nhưng 1 seed không đo được mức
nhiễu training (CI bootstrap chỉ đo bất định của tập test; sửa 15/11, trước đó ghi "nhiễu cỡ ±2 điểm" là suy diễn quá).
Bản exposed ngang bản công bố trên các nhóm lớn (man_none −1,0, other −0,6) nên pilot không phải bản train hỏng.
Giới hạn: 1 seed, 1 fold, B-L1 (scorer vẫn học nhãn PDM của proposal trên cảnh withheld). Purge rộng còn bỏ một phần
thành phần: VRU cắt qua khi không rẽ có 415 cửa sổ ở withheld, 481 ở exposed (audit
`output/reviews/cg_protocol_audit_20261115.json`); chiều ảnh hưởng là làm withheld kém đi (phóng to gap), nên không giải
thích được gap ≈ 0, nhưng nếu sau này thấy gap thì phải tách phần này ra.

### 10.2 Kiểm tra liều tiếp xúc (chốt 15/11 trước khi có kết quả)

Câu hỏi: kết quả "gap ≈ 0" ở §10.1 có ý nghĩa không, hay phép đo không nhạy với việc model có được train trên tổ hợp
hay không. Bản exposed chỉ thấy 755 cửa sổ của ô (0,9 % dữ liệu). Không cần GPU: chấm hai checkpoint pilot trên **cửa sổ
train** (`scripts/hgo/cg_dose.py`, `scripts/hgo/cg_dose_eval.sbatch`, metric cache gộp `exp/cg_drivor/train_metric_cache`).

- Nhóm: `ex_only_cell` = mọi cửa sổ ô chặt chỉ exposed được train; `ex_only_purged` = 500 cửa sổ khác trong vùng purge rộng
  (chỉ exposed); `wh_only` = cùng số cửa sổ với `ex_only_cell`, lấy trong các cửa sổ không đối tác chỉ withheld được train
  (kiểm tra ngược chiều); `both_<nhóm>` = 500 cửa sổ mỗi nhóm same_man / same_partner / man_none / other mà cả hai đều
  được train. Seed 0. Nhãn nhóm giống nhãn của cảnh test.
- Thống kê: d = exposed − withheld theo nhóm, bootstrap theo log và session; hiệu của hiệu d(ex_only_cell) − d(both_*),
  d(wh_only) − d(both_*).
- Cách đọc (chốt trước):
  - **A. Không nhạy**: d(ex_only_cell) ≈ 0 (CI chứa 0) và hiệu của hiệu với both_* chứa 0 → train trên chính cửa sổ đó
    không để lại dấu vết đo được bằng EPDMS / NC ở liều này; "gap ≈ 0" ở §10.1 **không** dùng được làm bằng chứng không có
    gap CG.
  - **B. Nhạy**: d(ex_only_cell) > 0 và hiệu của hiệu với both_* > 0 (CI không chứa 0), trong khi gap trên cảnh test của ô
    ≈ 0 → exposed học được điều gì đó trên cửa sổ đã thấy nhưng không mang sang cảnh ô mới; "gap ≈ 0" ở §10.1 có ý nghĩa
    (trong giới hạn B-L1, 1 seed, liều này).
  - wh_only chỉ là kiểm tra phụ: các cửa sổ không đối tác vốn dễ (điểm gần trần), d ≈ 0 ở đó không bác bỏ B.
  - Trường hợp lưng chừng (d > 0 nhưng hiệu của hiệu chứa 0): báo nguyên số, không kết luận.

## 11. Trạng thái

- [x] Nhãn tình huống, định nghĩa cắt qua + giao lộ, chia session, pool, fold, manifest, ảnh kiểm tra.
- [x] Metric cache + BEV target cho log mới (job 65301, 11 giờ 42 phút, 13/11): 64.700 / 65.412 frame train và
  4.148 / 4.189 frame test có cache (devkit bỏ ~1 %, có lẽ frame không có route).
- [x] Lọc frame train không có cache + cân bằng cặp (14/11, `scripts/hgo/cg_drivor_prep.py`): `train_<variant>_cached.txt`.
  HGO `train.py` vẫn đọc `train_<variant>.txt` và lọc lúc chạy: chuyển sang manifest chung trước run HGO.
- [x] Bước 0: DrivoR công bố trên các nhóm test (§9).
- [ ] Đo phủ cấu hình tại điểm xung đột (tiêu chí 5).
- [x] Nối vào train HGO (13/11): `--fold <dir> --fold-variant withheld / exposed / full`, nhiều thư mục metric cache,
  `--holdout-cell` (L2, `src/ade2e_wm/hgo/holdout.py`). Kiểm tra trên 4 log: mask L2 không đánh dấu quỹ đạo expert nào
  trong cửa sổ withheld (0 / 150); trong cửa sổ bị purge, 27 % quỹ đạo expert 4 s bị đánh dấu (phần còn lại bị purge do
  horizon 10 s hoặc do frame sau t0 trong cửa sổ).
- [x] Log val (13/11): `val_logs.txt`, 47 log của 24 session train không có cửa sổ nào trong pool (theo dõi, dừng warm-up,
  chọn ngưỡng); không thuộc session test CG hay navtest.
- [ ] Baseline đọc cùng manifest.

## File

| File | Nội dung |
|---|---|
| `scripts/hgo/cg_events.py` (+ `.sbatch`) | nhãn tình huống mọi frame |
| `scripts/hgo/cg_cells.py` (+ `.sbatch`) | đếm ô, purge, support, cỡ mẫu theo navtest → `docs/cg_split_candidates.md` |
| `scripts/hgo/cg_split.py` | chia session, pool, fold, manifest → `data/_lists/cg/split_v1/`, `docs/cg_split_folds.md`, scene filter `configs/navsim_cg/` |
| `scripts/hgo/cg_samples.py` (+ `.sbatch`) | ảnh sample |
| `scripts/hgo/cg_cache.sbatch` | BEV target + metric cache cho frame mới |
| `data/_lists/cg/split_v1/<fold>/` | `train_withheld.txt`, `train_exposed.txt`, `test_groups.csv` |
