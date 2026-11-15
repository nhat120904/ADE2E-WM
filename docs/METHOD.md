# HGO v3 — method

**Bản hiện hành, cập nhật 12/11/2026.** v3 là method duy nhất đang theo đuổi. Phần lõi (world model, scorer theo tác nhân, nhãn
theo cặp) đã có trong code và đã qua unit test; phần perception theo thời gian, vùng nhìn và hỗ trợ protocol CG **chưa code**
(§8). Chưa có run v3 nào được train.
Lý do của từng lựa chọn: [METHOD_RATIONALE.md](METHOD_RATIONALE.md). Thí nghiệm, bảng, baseline, compute:
[RESEARCH_DESIGN.md](RESEARCH_DESIGN.md). Split CG trên NAVSIM: [CG_SPLIT.md](CG_SPLIT.md). Bản cũ của các doc (H32, v2):
`exp/archive/docs_2026-11-12/`.

## 1. Bài toán và ý tưởng

Bài toán: planner lái xe từ camera phải xử lí được **tổ hợp tình huống chưa thấy khi train** (compositional generalization,
CG), ví dụ rẽ ở giao lộ khi có người đi bộ cắt qua, trong khi đã thấy riêng "rẽ" và riêng "người cắt qua".

Ý tưởng: điểm của một quỹ đạo theo metric lái xe là tích của các điều kiện **theo từng tác nhân** (không va chạm, TTC) và các
điều kiện **theo bối cảnh** (trong làn, đúng chiều, tiến độ). HGO tổ chức representation theo đúng cấu trúc đó:
- **register** giữ bối cảnh (đường, làn, giao lộ, vật ngoài vocabulary);
- **slot** neo từng tác nhân (vị trí, lớp, đặc trưng ảnh tại tác nhân, chuyển động);
- **world model** dự đoán tương lai của từng tác nhân và của bối cảnh bằng cơ chế dùng chung;
- **scorer** chấm mỗi cặp (quỹ đạo ứng viên, tác nhân) bằng cùng một hàm, chấm bối cảnh bằng hàm riêng, rồi **ghép theo dạng của
  metric**.

Tổ hợp mới khi đó được chấm bằng các phần đã thấy: "quỹ đạo rẽ" (term bối cảnh) và "quỹ đạo cắt tương lai của một người đi bộ"
(term theo cặp). Đây là giả thuyết cần kiểm chứng, không phải bảo đảm (METHOD_RATIONALE §3).

## 2. Input / output

| | NAVSIM (chính) | HUGSIM (zero-shot) | CARLA (khi có hạ tầng) |
|---|---|---|---|
| Camera | 4 camera F0, L0, R0, B0, 1148 × 672 | ánh xạ camera về 4 hướng | 4 camera tương ứng |
| Lịch sử | frame hiện tại + frame trước (0,5 s) cho perception; dịch chuyển 0,5 s của tác nhân qua ghép slot với detection t−1 | lấy mẫu về 2 Hz | 2 Hz |
| Trạng thái xe mình | vận tốc, gia tốc, lệnh điều hướng (trái / thẳng / phải) | lệnh ánh xạ lại | điểm route / lệnh |
| Không dùng lúc test | LiDAR, bản đồ, nhãn GT | | |
| Output | 64 quỹ đạo ứng viên (8 pose, 4 s, 2 Hz) và quỹ đạo được chọn | như NAVSIM | như NAVSIM |

Nhãn GT (box, track, bản đồ, điểm PDM) chỉ dùng làm giám sát lúc train. Input không bị ép giống baseline: mỗi method dùng cấu
hình của nó; cấu hình khớp nhau chỉ đặt ra trong ablation.

## 3. Pipeline

```
4 ảnh t, 4 ảnh t−0,5 s ─► DINOv2 ViT-S/14-reg4 + LoRA r32 (dùng chung)
  ├─ 16 scene token / camera ─► 64 register (bối cảnh)            [probe vai trò: chỉ lớp bản đồ]
  └─ patch t ⊕ patch t−1 (gộp 2 × 2, tia camera t−1 đặt trong hệ toạ độ xe lúc t) ─► head perception BEV
        ├─ CenterNet 4 lớp (xe, người đi bộ, xe đạp, vật tĩnh) + box + vận tốc ─► detection (ô 0,5 m, ngưỡng 0,3)
        │     └─ ≤ 32 slot, gần trước; slot = RoIAlign tại box ⊕ hình học ⊕ lớp ⊕ điểm  (detach)
        │           └─ ghép với detection của frame t−1 (cùng lớp, quanh vị trí vận tốc không đổi) ─► dịch chuyển 0,5 s của slot   [probe vai trò: lớp object]
        └─ BEV ngữ nghĩa (phụ)
state S = [64 register ; ≤ 32 slot] + token xe mình

world model (một lần mỗi cảnh, dùng chung trọng số cho mọi slot; không nhận hành động trên log-replay)
  mỗi slot k:  τ̂_k  forecast vật lý 8 pose (hệ toạ độ xe mình lúc t)
               ĥ_k  latent tương lai của tác nhân ở 1 / 2 / 4 s
  register:    R̂    register tương lai
  train: target JEPA = encoder EMA trên ảnh tương lai thật; head góc nhìn (chỉ train) đưa target về góc nhìn của xe log

generator    64 ứng viên p (decoder kiểu DrivoR trên [xe mình, S])
scorer       cho mỗi p, không attention giữa các ứng viên:
  bối cảnh   g_DAC, g_DDC, g_EP (p, R, R̂); comfort c(p, lịch sử xe mình); cờ vùng a(p) = g_area(p, R) (dự đoán)
  theo cặp   r_k(p) va chạm có lỗi, t_k(p) vi phạm TTC  ← hình học tương đối (p, τ̂_k), slot, lớp, kích thước, ĥ_k, a(p)
  ghép       NĈ(p) = Π_{tác nhân}(1 − r_k) × [0,5 + 0,5 Π_{vật tĩnh}(1 − r_k)];  TTĈ(p) = min_k (1 − t_k)
             Ŝ(p) = NĈ · σ(g_DAC) · σ(g_DDC) · (w_EP σ(g_EP) + w_TTC TTĈ + w_C σ(c)) / Σw
chọn         argmax_p Ŝ(p)
```

NĈ là kỳ vọng của điểm NC của PDM khi các rủi ro độc lập: PDM lấy min qua các va chạm (0 với tác nhân, 0,5 với vật không phải
tác nhân), nên nhiều va chạm vật tĩnh vẫn là 0,5 (unit test kiểm cả ba trường hợp).

## 4. Thành phần

| # | Thành phần | Thiết kế | Trạng thái |
|---|---|---|---|
| 1 | Encoder | DINOv2 ViT-S/14-reg4 (tự giám sát, không thấy dữ liệu lái) + LoRA r32; khởi tạo không từ checkpoint lái xe nào, để split CG sạch | có |
| 2 | Register | 16 scene token / camera = 64 register; neck → d = 256 | có (preset `V3`) |
| 3 | Ghép frame trước detection | 4 ảnh t−0,5 s qua cùng encoder (không lan gradient qua lượt này, `prev_grad=False`), lưới patch gộp trung bình 2 × 2, nhúng bằng tia của camera t−0,5 s đặt trong hệ toạ độ xe lúc t (bù chuyển động xe mình) + embedding "frame trước"; head BEV cross-attend [patch t ; patch t−1] | có (`prev_frame`) |
| 4 | Head perception | 40 × 32 BEV query (ô 2 m) cross-attend patch → upsample ×4 → lưới 160 × 128 ô 0,5 m, x ∈ [−32, 48] m, y ∈ [−32, 32] m; CenterNet 4 lớp, box, vận tốc; BEV ngữ nghĩa phụ (cache BEV chỉ có ±32 m ô 1 m: chỉ giám sát phần đó). Hai vật cách nhau ~1 m cho hai đỉnh (§7, có unit test) | có (`grid="v3"`) |
| 5 | Ngưỡng detection | 0,3, chọn trên val theo độ phủ tác nhân nguy hiểm (§7) | xong; đo lại sau khi đổi lưới |
| 6 | Slot | ≤ 32 detection hợp lệ, gần xe mình trước; RoIAlign tại box chiếu vào camera thấy rõ nhất + hình học, lớp, điểm; detach khỏi head detection | có |
| 7 | Lịch sử slot | head BEV giải mã riêng patch t−1 → detection t−1 trong hệ toạ độ lúc t (giám sát bằng box GT của t−1); mỗi slot ghép với detection t−1 gần nhất cùng lớp, trong 2 m quanh vị trí lùi theo vận tốc; dịch chuyển 0,5 s, đổi hướng, điểm → MLP cộng vào slot. **0,5 s thay cho 1,5 s** dự kiến: mỗi 0,5 s lịch sử thêm cần một lượt encoder nữa | có (`slot_hist`) |
| 8 | Giám sát vai trò | probe register chỉ giải mã lớp bản đồ; probe slot giải mã lớp object và box | có |
| 9 | World model | `StateWM`: một lần mỗi cảnh, mỗi token có query theo horizon, attention slot–slot và slot–register; τ̂_k (nhãn: tương lai GT của track khớp với slot), ĥ_k và R̂ (JEPA); head góc nhìn chỉ dùng khi train | có |
| 10 | Generator | 64 ứng viên, decoder kiểu DrivoR, 4 lần tinh chỉnh, min-of-N L1 | có |
| 11 | Scorer có cấu trúc | head bối cảnh + head theo cặp + phép ghép cố định như §3; không attention giữa ứng viên, nên chấm tập con ứng viên khi train vẫn nhất quán với lúc test | có |
| 12 | Readout đối chứng | trong **cùng** model: `mono` (một token / ứng viên đọc toàn bộ state và tương lai) và `pmono` (đọc đúng đặc trưng theo cặp của scorer có cấu trúc, ghép bằng attention học được). Cùng thông tin, chỉ khác cách ghép | có |
| 13 | Nhãn | mỗi bước, 16 ứng viên ngẫu nhiên được chấm online bằng logic PDM trên metric cache: va chạm có lỗi và TTC theo từng (ứng viên, track), EP từ progress thô, comfort theo lịch sử, cờ vùng; track → slot bằng ghép theo lớp, gate trước Hungarian | có |
| 14 | Train | stage 1 perception warm-up, stage 2 toàn bộ. Dừng warm-up (`--warmup-check 0.25`): mỗi 0,25 epoch đo recall slot (trung bình 4 lớp) trên log val; chuyển stage 2 khi tăng < 1 điểm (sau ít nhất 0,5, tối đa 2 epoch) | có |
| 15 | Hỗ trợ protocol CG | `--fold <dir> --fold-variant withheld / exposed / full`: log và token của pool, log val của split (47 log thuộc session train nhưng không có cửa sổ nào trong pool), thêm metric cache của split; `--metric-cache` nhận nhiều thư mục; `--holdout-cell "turn\|vru"` (L2): ứng viên rẽ và đi gần người đi bộ / xe đạp (PET < 3 s, định nghĩa rộng như purge) bị bỏ nhãn trong loss của mọi readout (`hgo/holdout.py`) | có |

**Hoãn, chưa đưa vào v3** (lý do ở METHOD_RATIONALE §2):
- 64 slot: ở ngưỡng 0,3 giới hạn 32 slot chỉ làm mất 0,4 % tác nhân nguy hiểm trên val (§7); đo lại sau khi đổi lưới;
- trọng số tồn tại e_k trong NĈ: dễ hạ rủi ro của người đi bộ thật nhưng điểm detection thấp;
- phần phản ứng theo ứng viên D(S, p) − D(S, p_ref): NAVSIM log-replay không có phản ứng; chỉ cần ở CARLA;
- free slot, competitive binding, map-element slot: ablation sau.

## 5. Tham số

Preset `V3` (đếm 13/11 trên code):
- trainable **36,65M** (V2 36,58M; thêm 32 register, MLP lịch sử slot, query BEV 40 × 32; encoder ViT đông cứng, chỉ LoRA
  train);
- khi chạy luật chọn có cấu trúc (bỏ bản EMA, probe, head góc nhìn, readout `mono` / `pmono`): **42,69M** (V2: 42,62M);
- state: 64 register + ≤ 32 slot = 96 token; bộ nhớ của head BEV: 15.744 patch frame t + 3.936 patch frame t−1 (đã gộp);
- DrivoR: 18,8M trainable / 41,8M tổng, cùng backbone.

Báo cả số train lẫn số lúc test cho mọi method.

## 6. Bằng chứng từ pilot dẫn tới v3

Từ run pilot H32 (32 register + ≤ 32 slot, scorer nguyên khối, 214 log, 1 seed; chi tiết trong bản archive):
- navtest v2.2 one-stage EPDMS 87,50 (DrivoR checkpoint công bố 88,75, khác dữ liệu train, chỉ để tham chiếu);
- **bộ chọn là nút thắt:** ứng viên tốt nhất trong 64 đạt 0,985 (PDM xấp xỉ), quỹ đạo được chọn 0,893; 99,8 % cảnh có ứng viên
  an toàn → scorer theo tác nhân, ghép theo metric (thành phần 11–13);
- **perception là giới hạn của scorer theo tác nhân:** trong các tác nhân gây nguy hiểm cho 64 ứng viên ở navtest, 69,5 % có
  slot, 16,5 % detector bỏ sót, 12,9 % ngoài ±32 m (gần như tất cả phía trước, 86 % trong 48 m), 1,2 % bị loại do giới hạn
  slot; bỏ sót người đi bộ 49 %, xe đạp 55 %, vật tĩnh 38 %, tăng theo khoảng cách → thành phần 3–5, 7, 14 (§7 tách lại trên
  val: phần lớn "bỏ sót" là hai vật gần nhau bị gộp thành một detection, hoặc lệch vị trí 2–4 m);
- H32 train với khung rẽ × VRU bị bỏ và chỉ 214 log, nên recall thấp có thể một phần do dữ liệu; v3 train trên pool 84k cửa sổ;
- split pilot chỉ đo "VRU ở gần", không đo tương tác → split mới theo xung đột cắt qua đường đi (CG_SPLIT.md).

## 7. Chẩn đoán perception (12/11/2026)

Checkpoint H32, 14.868 frame val (không phải navtest). Lưu 100 đỉnh heatmap trước ngưỡng mỗi frame. **Tác nhân nguy hiểm** =
track gây va chạm có lỗi hoặc vi phạm TTC với ít nhất 1 trong 64 ứng viên của model (nhãn PDM): 9.438 tác nhân. Mỗi tác nhân xếp
vào nguyên nhân đầu tiên đúng theo thứ tự trong bảng (cùng lớp, cổng 2 m, ghép một–một). Script: `scripts/hgo/dump_plans.py
--raw-det 100`, `scripts/hgo/perception_diag.py`; kết quả `exp/hgo/runs/H32_turn_vru_s0/perception_diag.json`.

| Nguyên nhân (%, ngưỡng 0,3) | Tất cả | Xe | Người đi bộ | Xe đạp | Vật tĩnh |
|---|---:|---:|---:|---:|---:|
| số tác nhân nguy hiểm | 9.438 | 5.282 | 1.376 | 266 | 2.514 |
| **có slot** | **66,0** | **89,0** | **31,0** | **50,0** | **38,3** |
| gộp với vật bên cạnh thành một detection | 12,4 | 0,4 | 35,7 | 0 | 26,1 |
| lệch vị trí 2–4 m | 7,7 | 1,2 | 18,2 | 48,5 | 11,2 |
| ngoài vùng ±32 m | 9,2 | 6,8 | 1,5 | 0 | 19,6 |
| sai lớp | 1,3 | 0,1 | 7,2 | 0 | 1,0 |
| dưới ngưỡng | 1,4 | 1,2 | 1,7 | 1,5 | 1,6 |
| không có đỉnh nào gần | 2,0 | 1,3 | 4,7 | 0 | 2,3 |

1. **Gộp đỉnh là lỗi lớn nhất của người đi bộ và vật tĩnh.** Có detection cùng lớp trên ngưỡng trong 2 m, nhưng nó đã được giao
   cho vật bên cạnh: 1.133 / 1.168 ca (87 % ca người đi bộ có một người nguy hiểm khác trong 2 m). Lý do: heatmap ô 1 m, lấy đỉnh
   bằng max-pool 3 × 3, nên hai vật cách nhau dưới khoảng 2 m chỉ cho một đỉnh. Nếu tính cả ca gộp, 67 % người đi bộ nguy hiểm có
   slot người đi bộ trong 2 m; nhưng một nhóm chỉ có một slot với box một người, nên scorer theo cặp không thấy người thứ hai.
2. **Giới hạn 32 slot chưa phải nút thắt:** chỉ 35 tác nhân (0,4 %) bị cắt do giới hạn.
3. **Ngưỡng 0,3 đã tốt nhất** (độ phủ 66,0 %). Hạ ngưỡng làm 32 slot đầy báo nhầm ở gần, mất xe ở xa: ngưỡng 0,05 thì 99 % cảnh
   đầy slot, độ phủ 58,9 %, recall xe ở 20–32 m từ 51,6 % xuống 10,2 %. Hạ ngưỡng cũng không tăng recall người đi bộ (~31 %).
4. **Ngoài vùng:** 873 tác nhân; 697 (80 %) ở phía trước trong 32–48 m, không ca nào phía sau. Vùng x ∈ [−32, 48] m lấy lại tối đa
   7,4 % tác nhân nguy hiểm, chủ yếu vật tĩnh và xe.
5. **Lệch vị trí:** người đi bộ 18 %. Số của xe đạp gần như chỉ từ một ca (129 frame, đều ở 12,4–12,5 m, có lẽ một người đi xe
   đạp đi cùng xe mình), nên chưa kết luận.
6. **Quỹ đạo được chọn:** 1,0 % cảnh có va chạm có lỗi. Trong 135 ca xác định được vật bị đâm, 70 % vật đó **có** trong state
   (scorer chọn sai dù thấy), 30 % không có (chủ yếu xe: 27 / 41). Perception giải thích khoảng một phần ba số lần chọn sai;
   scorer vẫn là nút thắt chính, khớp với §6.
7. **Ô giữ lại của pilot (rẽ × VRU):** độ phủ người đi bộ 31,8 % so với 31,0 % ở phần còn lại; quỹ đạo chọn có va chạm 1,5 % so
   với 1,0 %. Perception không kém hơn riêng ở ô giữ lại.

Giới hạn: H32 chỉ dùng frame hiện tại, train trên 214 log; tác nhân nguy hiểm định nghĩa theo 64 ứng viên của chính H32.

**Chốt cho v3:** ngưỡng 0,3; vùng x ∈ [−32, 48] m; heatmap và box ở ô **0,5 m** (160 × 128) để tách vật đứng gần nhau; giữ 32
slot nhưng đo lại sau hai thay đổi trên, vì số đỉnh sẽ tăng (với detection hoàn hảo, 32 vật gần nhất chỉ giữ 79,5 % tác nhân
nguy hiểm; nếu giới hạn bắt đầu cắt, chọn slot theo vùng xe mình tới được thay vì gần nhất). Frame trước giữ như thiết kế; chẩn
đoán này không đo trực tiếp được lợi ích của nó, nên đo bằng ablation (RESEARCH_DESIGN Bảng 4).

## 8. Việc còn lại để chạy được v3

1. ~~Chẩn đoán perception~~ xong (§7): ngưỡng 0,3, vùng tới 48 m, heatmap 0,5 m.
2. ~~Code v3~~ xong 13/11 (thành phần 2, 3, 4, 7, 14, 15; unit test `tests/hgo/test_v3.py`). Bản code trước khi sửa:
   `exp/archive/code_2026-11-13/`.
3. Smoke GPU (`scripts/hgo/smoke_v3.sbatch`): vài chục bước trên fold chính, đo thời gian / bộ nhớ mỗi bước, chạy thử
   `dump_plans.py` với frame trước.
4. Metric cache cho frame mới của split (job `cg_cache.sbatch`, đang chạy).
5. Run đầu: fold chính, bản exposed (toàn bộ pool, dùng chung cho Bảng 1) và bản withheld, 1 seed (RESEARCH_DESIGN §6).

## 9. Code

| File | Nội dung |
|---|---|
| `src/ade2e_wm/hgo/model.py` | `Config`, preset (`V3`, `V3pmono`, `V3mono`: cùng model, khác luật chọn), `Grid`; encoder, perception (`BEVDecoder`, `decode_boxes`, `select_slots`, `match_prev`), `SlotEncoder`, `StateWM`, generator, `StructuredScorer`, `structured_selection`, `MonoScorer`, `PairMonoScorer` |
| `src/ade2e_wm/hgo/holdout.py` | mask nhãn ứng viên của ô giữ lại (L2) |
| `src/ade2e_wm/hgo/pdm.py` | nhãn theo cặp từ logic PDM (`proposal_labels`) |
| `src/ade2e_wm/hgo/losses.py` | ghép slot–GT, loss detection, JEPA, góc nhìn, vật lý, scorer các readout |
| `src/ade2e_wm/hgo/data.py` | đọc log, cửa sổ, ảnh t và t−1 (`input_keys(prev)`), box GT theo vùng perception |
| `scripts/hgo/train.py`, `train.sbatch`, `smoke.sbatch`, `smoke_v3.sbatch` | train (`--fold`, `--holdout-cell`, `--cfg key=value` cho ablation), smoke |
| `scripts/hgo/dump_plans.py`, `dump.sbatch`, `eval_cpu.sbatch` | chạy model trên navtest / val, chấm EPDMS chính thức |
| `scripts/hgo/hazard_coverage.py`, `perception_diag.py`, `selection_analysis.py`, `oracle_props.py` | chẩn đoán |
| `scripts/hgo/cg_events.py`, `cg_cells.py`, `cg_split.py`, `cg_samples.py`, `cg_cache.sbatch` | split CG |
| `tests/hgo/` | unit test (30 test pass, 13/11; `test_v3.py`: lưới mới, frame trước, lịch sử slot, mask L2) |
