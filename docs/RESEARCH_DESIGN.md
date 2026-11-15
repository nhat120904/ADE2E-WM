# Thiết kế thí nghiệm: compositional generalization cho driving với hybrid predictive world state

**Cập nhật 12/11/2026.** Bài toán, protocol, bảng kết quả, baseline và kế hoạch compute cho HGO v3. Method:
[METHOD.md](METHOD.md); lý do và claim: [METHOD_RATIONALE.md](METHOD_RATIONALE.md); split CG: [CG_SPLIT.md](CG_SPLIT.md).

## 1. Bài toán, câu hỏi và claim

**Bài toán trung tâm: compositional generalization (CG) của planner lái xe.** Agent phải lái an toàn, có tiến độ khi gặp tổ hợp
tình huống chưa có trong training, dù các thành phần đã xuất hiện trong ngữ cảnh khác.

**Câu hỏi:** học và dùng một predictive world state thế nào để tái sử dụng kiến thức về object, bối cảnh và tương tác trên tổ
hợp chưa thấy, đồng thời giữ năng lực trên benchmark chuẩn?

| Lớp claim | Bằng chứng | Bảng |
|---|---|---|
| Agent lái được, không đánh đổi nhiều hiệu năng trong phân phối | benchmark chuẩn, đúng metric / protocol / regime dữ liệu | Bảng 1, 3 |
| Agent tổng quát hoá tốt hơn tới tổ hợp bị giữ lại | Δ_exposure và điểm tuyệt đối trên ô giữ lại, train lại mọi method trên cùng manifest | Bảng 2 |
| Thành phần nào tạo lợi ích | ablation state, world model, readout; đo trên ô giữ lại, không chỉ probe | Bảng 4 |

Contribution C1–C3 và danh sách không claim: METHOD_RATIONALE §5.

## 2. Bố cục thí nghiệm của các bài liên quan

| Primary source và phần đã đọc | Cách tổ chức bằng chứng / bài học áp dụng |
|---|---|
| [DrivoR, CVPR 2026](https://arxiv.org/html/2601.05083v2), §4.1–4.2, Tables 1–4 | Main tables NAVSIM-v1, NAVSIM-v2/NavHard và closed-loop HUGSIM; sau đó ablate representation, trajectory và scorer. Tách dữ liệu SimScale và bản metric đã sửa lỗi. Đây là mẫu bố cục benchmark → analysis phù hợp. |
| [WoTE, ICCV 2025](https://arxiv.org/html/2504.01941v2), Tables 1–3, §4.3–4.4 | Main NAVSIM + Bench2Drive; ablation phân biệt proposal-only, current-state evaluator và evaluator có future states. HGO tách lợi ích học representation khỏi việc dùng tương lai lúc suy luận. |
| [World4Drive, ICCV 2025](https://arxiv.org/html/2507.00603v1), §4.3–4.4 | Main benchmark trước, rồi component, condition slices và scaling. Night/rain/turn slices không tự thành strict compositional holdout: bài không loại các tổ hợp đó khỏi training ở thí nghiệm này. |
| [SVIB, NeurIPS 2023](https://arxiv.org/html/2311.09064v1), §2.2–4, §6 | Mọi primitive xuất hiện, giữ lại một số combinations, thay đổi training diversity với sample size cố định; báo ID, OOD và gap. Có optional omni-composition visual pretraining, nên exposure scope phải khai báo. |
| [HOWM, ICML 2022](https://arxiv.org/html/2204.13661v2), §4, §6, Appendix H | Object compositions train/test rời nhau nhưng mọi object có trong train; đánh giá multi-step prediction và resource scaling. Permutation đơn thuần không được coi là tổ hợp mới. |
| [COSMOS, ICLR 2024](https://arxiv.org/html/2310.12690v2), §2, §4, Appendix A.5 | Tách entity composition và relational composition; atom distributions được giữ nhất quán; đánh giá dynamics và downstream planning. Một số latent retrieval metrics tốt có thể đi kèm representation collapse. |
| [Are Object-Centric Representations Better at Compositional Generalization?, ICML 2026](https://arxiv.org/html/2602.16689v1), §3–4 | Kiểm soát training diversity, sample size, representation size, downstream capacity và FLOPs. Có compressed dense và k-means controls. HGO không thể chỉ dùng token budget để bảo đảm fairness. |
| [Provable Compositional Generalization for Object-Centric Learning, ICLR 2024](https://arxiv.org/html/2310.05327v2), §5–6 | Kết quả lý thuyết đòi hỏi các giả thiết decoder và consistency cụ thể. Không thể dùng định lý này làm bảo đảm cho transformer dynamics hoặc product risk head của HGO. |
| [Causal-JEPA, ICML 2026](https://arxiv.org/html/2602.11389v2), §5.2/Table 6, Appendix H | Chuỗi controls giữ nguồn DINO features, thay representation, objective và masking lần lượt. Áp dụng bằng việc tách state, predictive loss và planner-use của future; không mặc định thêm slots là đủ. |
| [DriveCombo, CVPR 2026](https://arxiv.org/html/2603.01637v1), §5.1, Appendix §9.1 | Có rule-composition evaluation và thử closed-loop 15 routes; split được mô tả là 80/20 samples, chưa đủ để suy ra held-out rule combinations. Dùng để tham khảo semantic scenarios, không sao chép làm strict split. |
| [Bench2Drive, NeurIPS 2024](https://arxiv.org/html/2406.03877v3), §3.2–3.3, §4 | 44 interaction scenarios và 220 evaluation routes; short routes giúp phân tích kỹ năng. Benchmark đa tình huống vẫn cần định nghĩa composition holdout riêng nếu dùng cho claim compositionality. |
| [NAVSIM v2 / Pseudo-Simulation](https://arxiv.org/html/2506.04218), §3 và [official metrics](https://github.com/autonomousvision/navsim/blob/v2.2/docs/metrics.md) | Phân biệt navtest one-stage, NavHard two-stage và metric version. Stage-two perturbations kiểm tra recovery/robustness, không tự chứng minh unseen combinations. |
| [INTERACT, arXiv 2611.02526](https://arxiv.org/abs/2611.02526), Tables 1–6, 10–14 | NAVSIM-v1 → NAVSIM-v2 navtest → nuScenes open-loop; rồi ablation tín hiệu WM, action conditioning, scale video pretraining; appendix: chất lượng slot, probe, dense vs slot alignment, scorer supervision. Không có test tổng quát hoá ngoài benchmark chuẩn. |
| [GraphWorld, arXiv 2606.16274](https://arxiv.org/html/2606.16274), Tables 1–11 | nuScenes 6 s / 3 s, Bench2Drive (open + closed loop, multi-ability), NAVSIM-v1, NAVSIM-v2 navhard và navtest; ba bảng robustness (slice khó / nhiễu, không giữ lại tổ hợp); một bảng ablation trên cả nuScenes, NAVSIM-v1 và Bench2Drive. |
| [World4Scorer, arXiv 2609.36438](https://github.com/World4Scorer/World4Scorer) (README) | NAVSIM-v2 navtest 91,4 (93,0 với re-ranking), NAVSIM-v1 94,0, Bench2Drive-220 closed loop, thêm OGBench-Cube ngoài driving. Code + weights đã công bố; NAVSIM agent xây trên code DrivoR, nên retrain được theo Protocol B. |
| [WA-JEPA, arXiv 2608.20974](https://github.com/AFARI-Research/WA-JEPA) (README, `docs/hugsim_closed_loop.md`) | NAVSIM v2.2 navtest 91,7 + HUGSIM zero-shot closed loop 436 scenario (HD-Score 0,4462). Repo có client HUGSIM cho model NAVSIM 4 camera và chấm lại baseline đã công bố theo cùng một protocol. |
| [Fail2Drive, arXiv 2604.08535](https://arxiv.org/html/2604.08535v1), Tables 1–2 | Benchmark paper: cặp route (trong phân phối vs đổi một yếu tố) trên CARLA, protocol Bench2Drive (DS, SR, trung bình điều hoà); 7 model (TCP, UniAD, Orion, HiP-AD, SimLingo, TF++, PlanT 2.0); bảng theo nhóm. Chỉ dùng để test. |
| [CausalAgents, arXiv 2207.03586](https://arxiv.org/abs/2207.03586), Tables 2–10 | Robustness benchmark cho motion forecasting: xoá agent không nhân quả, đo độ nhạy của nhiều model; rồi augmentation và thêm dữ liệu như cách sửa. |

Đây là tổng hợp thiết kế thí nghiệm từ các phần được đọc, không phải tuyên bố tất cả paper đều dùng cùng bố cục hoặc đều chứng minh strict compositional generalization.

Mẫu chung của các bài method AD ở trên (12/11): (1) bảng benchmark chuẩn đứng đầu (NAVSIM-v1 / v2, thường thêm nuScenes);
(2) một benchmark thứ hai, thường closed loop (Bench2Drive, HUGSIM, CARLA); (3) ablation từng thành phần (DrivoR làm trên navval);
(4) slice khó / robustness hoặc diagnostics; (5) hiệu năng (latency, params). Không bài method AD nào trong số này giữ lại tổ hợp
kèm exposed reference; thiết kế đó đến từ các bài CG trong ML (SVIB, HOWM, COSMOS, Kapl et al.). HGO giữ bố cục (1)–(5) và thêm
bảng Protocol B ngay sau phần benchmark chuẩn.

## 3. Protocol

### Protocol A — benchmark chuẩn

- **NAVSIM v2.2 navtest one-stage, EPDMS** (panel chính); **NavHard two-stage, EPDMS** với cấu hình traffic agent của benchmark;
  **NAVSIM v1 navtest, PDMS** để so với bảng v1. Mỗi panel một evaluator, không trộn số.
- **HUGSIM zero-shot closed loop** (436 scenario) qua client của WA-JEPA (`third_party/WA-JEPA/close_loop`), chấm lại các baseline
  theo cùng protocol.
- Chọn checkpoint / hyperparameter trên val tách log, không trên navtest / NavHard.
- **Regime dữ liệu:** model của Bảng 1 dùng chung với mốc exposed của Bảng 2, train trên pool CG = navtrain trừ 34 session test
  CG (≈ 77 % navtrain) cộng frame xung đột. Ghi rõ khi đặt cạnh số công bố (train trên toàn navtrain); số công bố để ở nhóm
  riêng, checkpoint công bố được chấm lại bằng cùng evaluator khi có (ví dụ DrivoR: 88,72 EPDMS với v2.2 trong project, README
  World4Scorer ghi 90,7 với bản evaluator khác).

### Protocol B — CG trên NAVSIM, exposure có kiểm soát

Chi tiết: CG_SPLIT.md. Tóm tắt:
- nhãn tình huống cho mọi frame: hành động (rẽ trái / phải **trong giao lộ**, đi thẳng, đường cong, dừng) × đối tác có quỹ đạo
  **cắt qua** đường đi của xe (người đi bộ / xe đạp, xe ngược chiều, xe cắt ngang), PET < 3 s, xe tới điểm xung đột trong 6 s;
- tách theo drive session: 137 session train (pool 84.390 cửa sổ), 34 session test CG phụ + navtest;
- **fold chính:** rẽ ở giao lộ × người đi bộ / xe đạp cắt qua: 241 cảnh test (32 navtest + 209 phụ), 50 log, 28 session;
  **fold tham khảo:** rẽ trái × xe ngược chiều (97 cảnh, support yếu);
- mỗi method train hai bản trên cùng pool: **withheld** (bỏ các cửa sổ có tổ hợp, kể cả khi tổ hợp chỉ nằm trong target tương
  lai, 3,1 % pool) và **exposed** (bỏ cùng số cửa sổ ngẫu nhiên không có đối tác); Δ_exposure = S(exposed) − S(withheld) trên
  cùng cảnh test;
- mức holdout: L1 (dữ liệu expert + target) cho mọi method; L2 (mask nhãn ứng viên tạo ra tổ hợp) là biến thể B-L12;
- mọi giai đoạn train dành riêng cho lái xe (warm-up perception, world model, scorer) tuân cùng manifest; encoder khởi tạo từ
  DINOv2 (không thấy dữ liệu lái), không từ checkpoint lái xe nào.

### Protocol C — tổ hợp có phản ứng (tạm gác)

CARLA chờ user xử lý hạ tầng (11/11). Khi có: bộ tổ hợp có kiểm soát (route × tác nhân × hành vi) với exposed reference; panel
ngoài Fail2Drive (chỉ test, protocol Bench2Drive). Không đổi tên NavHard thành benchmark CG.

## 4. Bảng và hình

### Bảng 1 — benchmark chuẩn

Panel: NAVSIM v2.2 navtest EPDMS (cùng 9 sub-score ở appendix), NavHard two-stage, NAVSIM v1 PDMS; HUGSIM ở Bảng 3. Cột:
method, input (camera / LiDAR, số frame lịch sử), backbone, dữ liệu train, giám sát perception, số ứng viên, EPDMS và sub-score
chính, params lúc test, latency cùng phần cứng. HGO là dòng cuối; không biến bảng này thành sweep biến thể.

### Bảng 2 — CG trên NAVSIM (bảng quyết định claim)

| Method | Params (test) | Seen ↑ | Held-out ↑ | Held-out khi được thấy | Δ exposure ↓ | Va chạm có lỗi ở held-out ↓ |
|---|---|---|---|---|---|---|
| LTF (TransFuser, camera) | | | | | | |
| DrivoR | 41,8M | | | | | |
| World4Scorer | | | | | | |
| WA-JEPA (nếu đủ GPU) | | | | | | |
| **HGO v3** | | | | | | |

- **Seen:** model withheld trên các nhóm cảnh test khác (cùng hành động không đối tác, cùng đối tác hành động khác, cảnh thường);
  cột này để chắc method không tụt ở cảnh bình thường.
- **Held-out:** model withheld trên 241 cảnh của fold chính. **Held-out khi được thấy:** model exposed trên cùng 241 cảnh.
- **Δ exposure** = (held-out khi được thấy) − (held-out) = số điểm mất chỉ vì chưa thấy tổ hợp.
- Claim: Δ của HGO nhỏ hơn và held-out cao hơn, trong khi seen không thua.
- Mỗi ô: trung bình ± độ lệch qua seed, CI bootstrap theo log cho hiệu HGO − baseline; caption ghi số cảnh / log. Fold tham khảo
  ở appendix. Thêm cột tỉ lệ tác nhân gây nguy hiểm có slot (với method có slot) để tách lỗi perception.

### Bảng 3 — closed loop / có phản ứng

HUGSIM zero-shot (HD-Score, các sub-metric) cho HGO và các baseline có client. CARLA (Protocol C) và Fail2Drive khi có hạ tầng.

### Bảng 4 — ablation (trên fold chính, báo seen và held-out)

| Nhóm | Hàng | Câu hỏi |
|---|---|---|
| Readout | mono → pair-mono → có cấu trúc (cùng model, chỉ đổi luật chọn) | phép ghép theo tác nhân có tạo lợi ích không |
| World model | không học tương lai / học nhưng đọc hiện tại / học và đọc; vận tốc không đổi; forecast hình học + kiểm tra va chạm | học và dùng tương lai đóng góp gì |
| State | register-only / slot-only / hybrid, cùng readout có thể dùng; thêm cấu hình cùng tổng token làm control năng lực | hybrid có giúp tái sử dụng hơn không |
| Perception | có / không frame trước; vùng ±32 m / x tới 48 m; heatmap ô 1 m / 0,5 m | lợi ích phụ thuộc độ phủ tác nhân ra sao |
| Protocol | B-L1 vs B-L12 | ứng viên phản thực có làm rò tổ hợp không |

### Hình và chẩn đoán

1. Sơ đồ protocol và ảnh sample của fold (`docs/figures/cg_samples/`).
2. Lưới support: số cửa sổ / log của từng ô và từng nguyên tử sau purge.
3. Oracle trên cùng tập ứng viên: điểm được chọn, tốt nhất, regret, phân biệt va chạm.
4. Độ phủ tác nhân nguy hiểm và recall perception theo lớp / khoảng cách / nhóm test.
5. Sai số forecast trên ô giữ lại theo horizon và lớp, so với vận tốc không đổi.
6. Cặp đã thấy → chưa thấy định tính, kèm ca thất bại.

## 5. Baseline

| Baseline | Input | Code | Cách dùng |
|---|---|---|---|
| DrivoR, CVPR 2026 | 4 camera, frame hiện tại | có (vendored `src/ade2e_wm/agents/drivor`) | cha trực tiếp; train lại trên manifest |
| World4Scorer, 2026 | như DrivoR | có + weight (NAVSIM agent trên code DrivoR) | đối thủ gần nhất (WM làm scorer, outcome nguyên khối); train lại, holdout áp cả lên nhãn ứng viên |
| LTF (Latent TransFuser) | camera | devkit NAVSIM | baseline E2E kinh điển; train lại |
| WA-JEPA, 2026 | 4 camera, 4 frame lịch sử | có (`third_party/WA-JEPA`) + client HUGSIM | train lại nếu đủ GPU; dùng client cho Bảng 3 |
| SafeDrive, CVPR 2026 | camera + LiDAR, 3 frame | có + checkpoint | gần nhất về đánh giá theo cặp; **cần LiDAR, chưa tải** (bảng nhóm riêng hoặc bỏ) |
| INTERACT, MORE, GraphWorld, SparseWAM | — | không có code train cho NAVSIM | chỉ trích số công bố ở Bảng 1 |

Quy tắc: mọi method train trên cùng manifest dữ liệu của từng fold; mỗi method giữ input và kiến trúc của nó; báo params, số
token / slot, history, latency. Cấu hình khớp nhau (cùng input, cùng tổng token) chỉ là control trong Bảng 4.

## 6. Compute

- Một run HGO trên pool ≈ 35 GPU-giờ (pilot: 38 giờ cho 91k cửa sổ × 10 epoch, H100, 1148 × 672). v3 thêm frame trước: đo lại
  trong smoke. Thời gian train của baseline chưa đo.
- Mỗi method trên fold chính cần 2 run: **exposed** (toàn pool; dùng chung làm model Bảng 1, ngân sách lệch 3,1 % so với một mốc
  exposed riêng) và **withheld**.
- Bản đầy đủ của Bảng 2 (5 method × 2 × 3 seed) > 1.000 GPU-giờ; ngân sách 180 GPU-giờ / tháng (còn khoảng 76 giờ trong tháng
  11). Thứ tự: HGO exposed + withheld, 1 seed (~70 giờ) → DrivoR, World4Scorer (tháng 12) → seed thêm, ablation chọn lọc.

## 7. Thứ tự triển khai

1. **Dữ liệu:** split v1 xong; metric cache cho frame mới (`scripts/hgo/cg_cache.sbatch`, đang chờ).
2. **Chẩn đoán perception** trên val: xong (METHOD §7) → ngưỡng 0,3, vùng tới 48 m, heatmap 0,5 m.
3. **Code v3**: xong 13/11, 30 unit test pass (METHOD.md §4, §8); còn smoke GPU đo thời gian bước.
4. **HGO trên fold chính:** exposed + withheld, 1 seed; kiểm tra chức năng (regret, va chạm, độ phủ) trước khi đọc Δ.
5. **Baseline** DrivoR, World4Scorer trên cùng manifest; script tính các cột Bảng 2.
6. Seed thêm, ablation Bảng 4, Bảng 1 đầy đủ, HUGSIM.

## 8. Tiêu chí đóng claim

- Bảng chuẩn đủ coverage, ghi rõ metric / phiên bản / regime dữ liệu; hiệu năng trong phân phối bảo vệ được.
- HGO cải thiện điểm tuyệt đối và Δ_exposure trên ô giữ lại so với baseline mạnh, nhiều seed, CI theo log.
- Thành phần của mỗi tổ hợp giữ lại còn support; mọi giai đoạn train sạch exposure theo protocol đã công bố.
- State, objective dự đoán, việc dùng tương lai và readout có ablation riêng, đo trên planning ở ô giữ lại.
- Chẩn đoán (độ phủ, forecast, regret) giải thích được ít nhất một cơ chế của lợi ích.
