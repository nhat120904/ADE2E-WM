# HGO v3 — lý do thiết kế cho compositional generalization trong driving

**Cập nhật 12/11/2026.** Tài liệu trả lời: mỗi thành phần của v3 xuất phát từ yêu cầu nào của bài toán, vì sao tổ hợp đó có thể
giúp CG, điểm non-trivial, novelty so với các bài driving gần nhất, và rủi ro. Method và trạng thái code: [METHOD.md](METHOD.md);
thí nghiệm: [RESEARCH_DESIGN.md](RESEARCH_DESIGN.md); split: [CG_SPLIT.md](CG_SPLIT.md). Tiêu chí novelty theo user: **chưa
xuất hiện trong autonomous driving**. Mức kiểm chứng nguồn ghi rõ: *code* (đã đọc mã nguồn), *full text*, *abstract*. Số liệu
của các bài khác là số tác giả báo cáo, chưa tái lập.

## 0. Luận điểm

Hàm mục tiêu của lái xe có cấu trúc tổ hợp: an toàn là phép AND của các điều kiện theo **từng tác nhân**, còn tuân thủ và tiến
độ là các điều kiện theo **bối cảnh**. Một planner có thể tổng quát hoá sang tổ hợp (bối cảnh × tác nhân) chưa thấy khi:
1. trạng thái được tách thành phần bối cảnh (register) và phần từng tác nhân (slot);
2. tương lai được dự đoán theo từng tác nhân bởi một world model dùng chung trọng số;
3. điểm của quỹ đạo được ghép từ các term theo cặp và term bối cảnh bằng phép ghép **cùng dạng với metric**.

Khi đó cảnh "rẽ ở giao lộ + người đi bộ cắt qua" được chấm bằng các phần đã thấy: hàm bối cảnh "rẽ" và hàm rủi ro "quỹ đạo xe
mình so với tương lai của một người đi bộ", trong hệ toạ độ tương đối. Thiết kế **khuyến khích** tái sử dụng cơ chế; nó không
bảo đảm CG: đầu vào của world model ở tổ hợp mới vẫn có thể mới, và readout có cấu trúc không sửa được tác nhân bị bỏ sót hay
forecast sai. Đây là giả thuyết cần kiểm chứng (§4).

## 1. Cấu trúc của metric

Theo code PDM (`third_party/navsim_v2/.../scoring/pdm_scorer.py`, *code*):
- **NC** (dòng 363–420): duyệt từng thời điểm và từng cặp (quỹ đạo, track), lấy `np.minimum`. Va chạm **có lỗi** khi đâm trực
  diện lúc đang chạy, đâm vào track đang đứng yên, hoặc va chạm bên hông khi xe mình chiếm nhiều làn / ra ngoài vùng lái. Điểm 0
  nếu track là tác nhân (xe, người đi bộ, xe đạp; kể cả xe đứng yên), 0,5 nếu là vật không phải tác nhân. Nhãn "có lỗi" cần cả
  thông tin làn và vùng lái của xe mình.
- **TTC** (dòng 492–580): luật đạt / không đạt (chiếu footprint 1 s theo vận tốc không đổi), lấy min theo cặp, tức AND.
- **DAC, DDC, lane keeping**: chỉ phụ thuộc quỹ đạo và bản đồ.
- **EP** chính thức: progress thô chuẩn hoá theo max trên tập được chấm; **history comfort** ghép quỹ đạo quá khứ thật.
- Điểm cuối = tích các metric nhân × tổng có trọng số (EP, TTC, LK, comfort).

$$
S(p)=\mathrm{NC}(p)\cdot \mathrm{DAC}(p,c)\cdot\mathrm{DDC}(p,c)\cdot
\frac{w_{EP}\mathrm{EP}(p,c)+w_{TTC}\min_k\mathrm{TTC}(p,\tau_k)+w_{LK}\mathrm{LK}(p,c)+w_C\,\mathrm{C}(p)}{\sum w},\quad
\mathrm{NC}(p)=\min_k \mathrm{nc}(p,\tau_k)
$$

τ_k là tương lai của tác nhân k. NAVSIM navtest one-stage là log-replay: τ_k không phụ thuộc *p*. NavHard two-stage có IDM
phản ứng cho **xe**; người đi bộ và vật tĩnh vẫn theo log. Trong CARLA, τ_k = τ_k(p, c, E).

**Hệ quả:** hàm cần học tách được thành **term theo cặp (xe mình, tác nhân k)** và **term theo bối cảnh**. Một scorer nguyên
khối F(z(c, E), p), với z trộn bối cảnh và tác nhân, thấy z(c\*, E\*) như một đầu vào ngoài phân phối. Đây là giả thuyết, không
phải định lý: scorer nguyên khối vẫn có thể học được cách tổng quát hoá, và đối chứng phải kiểm tra đúng điều đó.

Kỳ vọng của NC khi các rủi ro độc lập: E[NC] = Π_tác nhân(1 − r_k) × [0,5 + 0,5 Π_vật tĩnh(1 − r_k)]. Đây là phép ghép của
scorer v3; nó khớp PDM ở các trường hợp tất định (hai vật tĩnh → 0,5; vật tĩnh + tác nhân → 0). Tích các xác suất biên chỉ bằng
xác suất chung khi rủi ro độc lập, nên cần kiểm tra calibration theo số slot.

## 2. Từ bài toán tới thiết kế

Nguyên tắc: mọi lựa chọn xuất phát từ bài toán; không chép input hay ngân sách token của baseline để "công bằng". Các bảng chuẩn
so mỗi method với cấu hình riêng của nó; Δ_exposure đo bên trong từng method; giữ input / ngân sách bằng nhau chỉ là việc của
ablation.

| Yêu cầu từ bài toán | Thiết kế v3 | Đã cân nhắc và không chọn |
|---|---|---|
| Term theo cặp cần **mỗi tác nhân một token** (cá thể hoá), biết **loại** tác nhân, và không bỏ sót tác nhân nhỏ. Navtest: median 6 xe trong 32 m, 24 % cảnh ≥ 12 xe, chưa kể người đi bộ, xe đạp, vật cản | slot neo bằng detection có giám sát (tồn tại, vị trí, lớp); nội dung slot = RoIAlign tại box + hình học, học tiếp bằng dự đoán tương lai và hệ quả | slot không giám sát kiểu INTERACT: 7 slot mức vùng, "nearby vehicles may share a slot" (*full text*), không có kiểu, dễ gộp người đi bộ nhỏ vào nền |
| Term bối cảnh cần bố cục đường, làn, giao lộ, vật ngoài vocabulary | register (scene token trong ViT như DrivoR), giám sát vai trò bằng probe chỉ giải mã lớp bản đồ | chỉ dùng slot vùng / map slot: là ablation, không phải mặc định |
| Không có **đường tắt** cho term an toàn qua token toàn cục | định tuyến ở readout: head theo cặp không đọc register; bối cảnh vào term theo cặp chỉ qua world model và cờ vùng a(p) (dự đoán từ register); head bối cảnh không đọc slot | cấm tuyệt đối thông tin chéo: không làm được (backbone attention toàn ảnh) và không nên (bối cảnh cần cho dự đoán người đi bộ) |
| Rủi ro phụ thuộc **tương lai** của tác nhân, tức phụ thuộc chuyển động; vận tốc đoán từ một ảnh kém tin cậy | dùng thêm frame trước: hợp nhất đặc trưng **trước** head detection (tăng recall vật đang chuyển động) và ghép slot với detection của frame trước (dịch chuyển 0,5 s; 1,5 s cần thêm hai lượt encoder, để ablation sau) | chỉ ghép slot sau detection: không đưa được tác nhân bị bỏ sót ở frame hiện tại vào state |
| Tác nhân gây nguy hiểm nằm ngoài vùng nhìn: trên val, 9,2 % tác nhân nguy hiểm ở ngoài ±32 m, 80 % số đó ở phía trước trong 32–48 m (METHOD §7) | vùng perception x ∈ [−32, 48] m | giữ 64 ô cho 80 m (1,25 m / ô): khó định vị người đi bộ |
| Mỗi người đi bộ trong một nhóm cần slot riêng: trên val, 36 % người đi bộ nguy hiểm và 26 % vật tĩnh nguy hiểm bị gộp với vật bên cạnh thành một detection (METHOD §7) | heatmap và box ở ô 0,5 m (160 × 128) | hạ ngưỡng: không tăng độ phủ (đã đo); detector kiểu query (DETR): thay đổi lớn, giữ làm phương án nếu ô 0,5 m chưa đủ |
| Bối cảnh không yếu hơn baseline | 16 register / camera (64); ablation DrivoR 8 → 16 / camera gần như không đổi (89,7 → 90,0 navval), nên chọn theo hiệu năng | tổng state cố định 64 token: là ràng buộc của brief, không xuất phát từ bài toán; chuyển thành ablation |
| Tương lai của tác nhân cho mọi ứng viên, rẻ, không causal confusion | world model chạy **một lần mỗi cảnh**, không nhận hành động trên log-replay (đúng với evaluator); hình học tương đối với từng ứng viên tính bằng công thức | WM chạy theo từng ứng viên (pilot H32): đắt gấp 64 lần, học được "xe mình phanh thì xe trước phanh" từ log |
| Latent tương lai mang thông tin ngoài box (tư thế, ý định) | ĥ_k học bằng JEPA (encoder EMA trên ảnh tương lai thật); head góc nhìn chỉ dùng khi train, để ĥ_k ít phụ thuộc góc nhìn của xe log | chỉ forecast vị trí: là đối chứng (vận tốc không đổi, forecast hình học + kiểm tra va chạm) |
| Cấu trúc tổ hợp đặt ở bộ đánh giá, generator chỉ cần phủ | propose–evaluate: 64 ứng viên; pilot: tốt nhất trong 64 đạt 0,985, được chọn 0,893 | hồi quy thẳng một quỹ đạo: không đặt được phép ghép |
| Term theo cặp cần nhãn **của đúng tác nhân**, và support rộng hơn hành vi trong log | nhãn PDM theo từng (ứng viên, track) cho ứng viên phản thực, tính online; track → slot bằng ghép theo lớp | nhãn chồng footprint (pilot): lệch định nghĩa at-fault |
| Kiểm tra được phần nào tạo lợi ích | ba readout trong **cùng** model, cùng thông tin: mono → pair-mono → có cấu trúc; chỉ đổi luật chọn | train riêng từng readout: trộn readout với lượng giám sát của representation |

**Hoãn:**
- 64 slot: ở ngưỡng 0,3 giới hạn 32 slot chỉ cắt 0,4 % tác nhân nguy hiểm trên val (METHOD §7). Nhưng với detection hoàn hảo,
  chọn 32 / 64 object gần nhất giữ 79,5 % / 86,1 % tác nhân nguy hiểm, nên phải đo lại sau khi đổi lưới (thêm vùng 48 m, ô
  0,5 m làm tăng số đỉnh); có thể chọn slot theo vùng xe mình tới được thay vì gần nhất.
- Trọng số tồn tại e_k (r_k → e_k r_k): ghép đúng về xác suất, nhưng điểm detection thấp của một người đi bộ thật sẽ hạ rủi ro
  đúng ở ca nguy hiểm nhất. Chỉ thêm khi đã có detection hiệu chỉnh trên val và đo được tác dụng.
- Phản ứng theo ứng viên D(S, p) − D(S, p_ref), p_ref = π_ref(S_t): chỉ học được khi dữ liệu xác định được nó (CARLA rẽ nhánh);
  trên log-replay bằng 0.

## 3. Vì sao có thể giúp CG

**Phát biểu (inductive bias, không phải định lý).** Nếu:
- (A) giao diện có vai: term theo cặp chỉ nhận bối cảnh qua dynamics và cờ a(p); term bối cảnh không đọc slot;
- (B) world model, r, t, g dùng chung và nhận đầu vào tương đối;
- (C) mỗi term được train trên support phủ cấu hình xuất hiện ở test (ứng viên phản thực mở rộng support);
- (D) phép ghép cùng dạng với metric,

thì các term ở tổ hợp mới có thể nằm trong phân phối, và cần ít dữ liệu hơn cho tổ hợp mới. Cơ sở: Wiedemer et al. (NeurIPS
2023, *full text*, Thm. 1: model dùng chung composition function với quá trình sinh dữ liệu và có compositional support thì
tổng quát hoá; là điều kiện **đủ**, kèm giả thiết factorization, độ trơn, khả năng xác định); decoder cộng (Wiedemer et al.,
ICLR 2024; Lachapelle et al., NeurIPS 2023); cơ chế dùng chung gắn với từng entity (HOWM, ICML 2022; LPWM, ICLR 2026); object-
centric hơn dense ở CG khó khi dữ liệu / compute hạn chế (Kapl et al., ICML 2026). Các định lý có giả thiết riêng, không áp
thẳng cho transformer dynamics hay head rủi ro; chúng là lý do chọn inductive bias, bằng chứng phải là thực nghiệm.

**Ví dụ "rẽ ở giao lộ + người đi bộ cắt qua":**
1. g_DAC / g_EP chấm các ứng viên rẽ trên register, như mọi cảnh rẽ đã thấy.
2. Slot người đi bộ giống người đi bộ ở cảnh đi thẳng; world model dự báo người đó đi tiếp qua vạch, **nếu** nó tổng quát hoá
   sang kiểu giao lộ này.
3. Với mỗi ứng viên, r chấm cấu hình tương đối "xe mình cắt đường đi của người đi bộ ở t = 2 s", đã gặp trong ứng viên phản
   thực ở ô khác (đi thẳng × người cắt qua).
4. Ứng viên nhường người đi bộ thắng.

**Chỗ dễ gãy:**
- **perception bỏ sót tác nhân**: (A) gãy ngay từ đầu. Đo trên pilot: người đi bộ gây nguy hiểm bị bỏ sót 49 %, xe đạp 55 %, vật
  tĩnh 38 %; 12,9 % tác nhân nguy hiểm ngoài vùng ±32 m. Đây là lý do của các thay đổi perception trong v3;
- backbone toàn cục làm slot mang bối cảnh; scorer có đường tắt qua register;
- tương tác bậc cao (người đi bộ dừng vì một xe khác) mà world model không học được;
- world model gặp đầu vào mới ở tổ hợp mới và dự báo sai; cờ a(p) dự đoán sai;
- generator không phủ maneuver cần thiết (đo oracle trên ô giữ lại cho mỗi run);
- tương tác có phản ứng (NAVSIM log-replay không có).

**Dự đoán kiểm chứng được:**
1. Readout có cấu trúc hơn pair-mono và mono **ở ô giữ lại** (Δ_exposure nhỏ hơn), trong khi cảnh đã thấy tương đương.
2. Lợi ích tập trung ở NC / TTC, không ở DAC / EP.
3. Chèn / xoá một tác nhân chỉ đổi trực tiếp điểm của các ứng viên giao với tác nhân đó.
4. Cho register vào thẳng head theo cặp thì lợi ích giảm (không bắt buộc đúng: bối cảnh có thể giúp).
5. Lỗi tập trung ở cấu hình tương đối nằm ngoài support (đo trước khi train) và ở tác nhân không có slot.
6. Sai số forecast trên tổ hợp mới báo riêng; lợi ích planning phải đi kèm forecast không tệ hơn.

## 4. Bốn phép thử quyết định

1. **Readout:** mono → pair-mono → có cấu trúc, cùng state, forecast, giám sát và tập ứng viên, trên ô giữ lại. Phép thử chính:
   lợi ích còn hay không khi chỉ đổi cách ghép. Thêm: SafeDrive bật / tắt term theo cặp (code của người khác).
2. **Dùng tương lai:** không học tương lai / học nhưng chỉ đọc hiện tại / học và đọc; vận tốc không đổi; forecast hình học +
   kiểm tra va chạm. Cùng planner.
3. **Exposure có kiểm soát:** split theo xung đột **cắt qua** đường đi và rẽ **trong giao lộ**, tách theo drive session, purge cả
   target tương lai (L1), tuỳ chọn mask nhãn ứng viên tạo ra ô giữ lại (L2); mốc exposed cùng ngân sách; Δ_exposure trong từng
   method (CG_SPLIT.md). Ứng viên phản thực có thể là support hợp lệ hoặc làm rò tổ hợp: so B-L1 với B-L12.
4. **Độ phủ tác nhân:** báo recall theo lớp / khoảng cách và tỉ lệ tác nhân gây nguy hiểm có slot trên chính cảnh test; tách lỗi
   do perception khỏi lỗi do scorer.

Nếu không thấy Δ_exposure, chỉ kết luận được là **chưa phát hiện** khoảng hụt với model, budget, metric và cỡ mẫu đang dùng.

## 5. Contribution và claim

**Câu hỏi:** học và dùng world state thế nào để planner dùng lại được kiến thức về tương tác khi gặp tổ hợp giao thông chưa
thấy? **Claim trung tâm (cần kiểm chứng):** giữ cấu trúc bối cảnh–thực thể xuyên suốt từ học state, qua dự đoán, tới đánh giá
hành động giúp planner tái sử dụng các tương tác đã học trên tổ hợp mới. Không dùng "lần đầu tiên".

**C1 — Method (HGO v3).** Planner từ camera học world state dự đoán dạng hybrid: register giữ bối cảnh, slot neo từng thực thể,
giao diện tách vai; world model dự đoán diễn biến vật lý và latent của từng thực thể bằng cơ chế dùng chung; mỗi cặp (ứng viên,
thực thể) được đánh giá bằng cùng một hàm rồi ghép với tiêu chí bối cảnh theo dạng của metric; nhãn theo cặp gắn rủi ro với
đúng thực thể. Representation và đánh giá là **một** đóng góp: hybrid là cách tổ chức representation để đánh giá theo thực thể
dùng lại được. Khác với INTERACT (slot vùng, decoder chung), World4Scorer (state theo ứng viên, outcome nguyên khối), SafeDrive
(thế giới riêng cho từng ứng viên, điểm chính từ head chung, term theo cặp tuỳ chọn), DTPP / DIPP (input vector), PrismAD (nhóm
token ngữ nghĩa).

**C2 — Protocol CG ở mức quyết định lái.** Tổ hợp hành động × tương tác bị giữ lại khỏi train; xung đột định nghĩa bằng quỹ đạo
cắt qua, kiểm bằng ảnh; tách theo drive session, thêm session test phụ vì navtest không đủ cảnh; purge target tương lai; mốc
exposed cùng ngân sách; báo an toàn, tiến độ, điểm tổng kèm độ bất định theo log và seed. Gọi là protocol cho tới khi đủ dữ liệu,
công cụ và baseline để gọi là benchmark.

**C3 — Phát hiện về nguồn gốc và giới hạn của lợi ích CG.**

| Claim | Vai trò | Bằng chứng |
|---|---|---|
| HGO cải thiện planning trên tổ hợp chưa thấy, giữ năng lực trên cảnh đã thấy | trung tâm | Δ_exposure và điểm tuyệt đối trên ô giữ lại, nhiều seed; benchmark chuẩn |
| Hybrid state giúp tái sử dụng tốt hơn state thay thế | representation | register-only, entity-only, hybrid dưới cùng readout; đo ở planning trên ô giữ lại |
| Học dự đoán và dùng tương lai lúc suy luận đóng góp riêng | world model | phép thử 2 |
| Đánh giá theo thực thể + phép ghép giúp chọn hành động | quyết định | phép thử 1 |
| Lợi ích phụ thuộc độ phủ thực thể và chất lượng forecast | giới hạn | độ phủ tác nhân nguy hiểm, sai số forecast, regret theo nhóm; B-L1 vs B-L12 |

**Không claim:** bảo đảm CG; state disentangled / causal; sample-efficient (trừ khi có đường cong dữ liệu); mô hình hoá phản ứng
(cần CARLA); SOTA NAVSIM; một ngân sách token cố định là tối ưu.

**Phương án theo kết quả:** C1–C3 đều đúng → paper method kèm phân tích; chỉ C2 + một phần C3 → paper phân tích / protocol;
Δ ≈ 0 với mọi method trên NAVSIM → báo phát hiện đó và chuyển bằng chứng chính sang mô phỏng.

## 6. Novelty trong autonomous driving (tính đến 11/2026)

Ký hiệu: ✓ có, ✗ không, ~ một phần.

| Công trình | Token tác nhân | Token bối cảnh có vai riêng | WM theo tác nhân | Tương lai theo ứng viên | Readout ghép theo tác nhân | Đánh giá CG có kiểm soát |
|---|---|---|---|---|---|---|
| TransFuser (NAVSIM) *code* | ✓ 30 agent query | ~ 64 token BEV, không tách vai | ✗ | ✗ | ✗ | ✗ |
| SparseDrive, ICRA 2025 *abstract* | ✓ instance + map | ~ map instance | ✗ | ✗ | ~ rescore va chạm hình học, không học | ✗ |
| WoTE, ICCV 2025 *code + full text* | ~ agent query (aux) | ✗ BEV dày | ✗ | ✓ 256 anchor | ✗ reward trên BEV | ✗ |
| DA-WAM, 2026 *full text* | ✗ | ✗ | ✗ | ✓ | ✗ scorer nhiều head | ✗ |
| GraphWorld, 2026 *full text* | ✓ instance SparseDrive, K hàng xóm | ✗ | ~ latent theo agent | ✗ | ✗ | ✗ |
| **INTERACT**, 11/2026 *full text* | ✗ 7 slot vùng | ✗ | ✓ slot + latent ego action | ~ | ✗ confidence head chung | ✗ |
| CarFormer / PlanT 2.0 *abstract* | ✓ (BEV GT) | ✗ | ~ | ✗ | ✗ | ~ thử nhiễu |
| RiskWorld, 2026 *full text* | ✓ object từ box có sẵn | ✗ | ✓ RSSM theo object | ✗ | ✓ rủi ro theo object, không lập kế hoạch | ✗ |
| DIPP / DTPP *chưa đọc full text* | ✓ agent (vector) | ✓ map | ✓ | DTPP ✓ | ✓ cost theo agent | ✗ |
| Hydra-MDP, 2024 | ✗ | ✗ | ✗ | ✗ | ~ ghép head sub-score theo PDM, không theo tác nhân | ✗ |
| SafeDrive, CVPR 2026 *code* | ✓ top-20 instance | ✗ | ✓ thế giới thưa theo ứng viên | ✓ | ~ head va chạm theo cặp, nhãn PDM theo tác nhân; cộng vào điểm tuỳ chọn (mặc định 0) | ✗ |
| World4Scorer, 2026 *code* | ✗ | ✗ | ✗ predictor chung | ✓ | ✗ outcome nguyên khối | ✗ |
| **HGO v3** | ✓ slot neo detection | ✓ giám sát vai + định tuyến | ✓ dùng chung | ✓ hình học tương đối | ✓ cùng dạng metric | ✓ |

Đã có ở đâu đó: WM trên object (INTERACT, CarFormer, RiskWorld, FLAM); tương lai theo ứng viên (WoTE, INTERACT, DTPP, SafeDrive);
dự đoán theo agent rồi ghép chi phí (DIPP, DTPP, NMP / P3 / MP3, PDM-Closed); nhãn PDM theo tác nhân và tích qua tác nhân
(SafeDrive); register dự đoán cho driving (MORE, SparseWAM); nhóm token tách vai (PrismAD); WM làm scorer với nhãn outcome cho mọi
ứng viên (World4Scorer); CG của mô hình sinh video driving (Farid et al., 2025). Register cùng slot trong object-centric learning
có ở CODA (ICLR 2026), ngoài AD.

Điểm khác của HGO: (i) register + slot từng tác nhân trong **một planner AD**, với giao diện tách vai đi tới tận phép ghép quyết
định, học từ camera; (ii) **bằng chứng CG có kiểm soát exposure** cho planner E2E được train lại. Chưa thấy bài AD nào có cả hai
(phạm vi: các bài trên, các PDF trong `ade2e-paper/`, tìm kiếm web 12/11).

## 7. Bài học cho cộng đồng

Nếu kết quả đúng dự đoán: có token object không tự tạo ra tính tổ hợp; tính tổ hợp là thuộc tính của **đồ thị tính toán** từ
state đến quyết định; metric lái xe gợi ý cách tách (an toàn theo tác nhân × tuân thủ theo bối cảnh); giá trị của world model cho
CG là tương lai theo từng thực thể, phải hơn forecast hình học; đo CG cần exposure có kiểm soát, mốc exposed và nhóm cha.

Nếu không đúng, vẫn có bài học: benchmark non-reactive có thể không đòi tính tổ hợp; nút thắt có thể nằm ở perception; hoặc slot
từ backbone toàn cục không đủ cục bộ.

## 8. Rủi ro và câu hỏi mở

- **Recall perception** là giới hạn trên của mọi lợi ích theo tác nhân; báo theo lớp, khoảng cách và nhóm test.
- **Cỡ mẫu:** tổ hợp thật hiếm trong log thực; fold chính có 241 cảnh test từ 50 log; MDE còn là ước lượng thô.
- **Support:** VRU cắt qua khi xe không rẽ chỉ có 432 cửa sổ chặt trong pool.
- **Calibration khi ghép nhiều slot:** báo nhầm cộng dồn qua phép tích làm xe phanh thừa, giảm EP.
- **Rò qua ứng viên phản thực:** so B-L1 với B-L12.
- **Giao diện cờ bối cảnh:** cờ dự đoán sai truyền sang luật at-fault và TTC.
- **Surrogate:** scorer v3 không khớp nhãn kiểu DrivoR (EP thô, at-fault theo cặp); so sánh phải nêu rõ.
- **Giả định không phản ứng** chỉ đúng với log-replay; claim về phản ứng cần CARLA.
- **Compute:** một run ≈ 35 GPU-giờ; ngân sách 180 GPU-giờ / tháng giới hạn số seed và baseline (RESEARCH_DESIGN §6).

## Nguồn

- TransFuser (NAVSIM v2 devkit, mã nguồn local); [WoTE](https://arxiv.org/abs/2504.01941), code [liyingyanUCAS/WoTE](https://github.com/liyingyanUCAS/WoTE)
- [LAW](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6aa4967920e495e90aeeaa3acf18d019-Abstract-Conference.html), [SSR](https://arxiv.org/html/2409.18341v2), [World4Drive](https://arxiv.org/abs/2507.00603), [SeerDrive](https://arxiv.org/abs/2510.11092)
- [SafeDrive](https://arxiv.org/abs/2602.18887) (code [SPA-junghokim/SafeDrive](https://github.com/SPA-junghokim/SafeDrive)), [PrismAD](https://arxiv.org/pdf/2607.10336), [World4Scorer](https://arxiv.org/html/2609.36438) (code [World4Scorer/World4Scorer](https://github.com/World4Scorer/World4Scorer)), [MORE](https://arxiv.org/html/2610.14871), [SparseWAM](https://arxiv.org/html/2611.00044), [FLAM](https://arxiv.org/html/2602.16229), [Farid et al.](https://arxiv.org/html/2510.03075), [TOAD](https://arxiv.org/pdf/2606.07170), [WA-JEPA](https://github.com/AFARI-Research/WA-JEPA)
- [INTERACT](https://arxiv.org/abs/2611.02526), [GraphWorld](https://arxiv.org/html/2606.16274), [RiskWorld](https://arxiv.org/html/2608.21414)
- [GenAD](https://arxiv.org/pdf/2402.11502), [SparseDrive](https://arxiv.org/html/2405.19620v2), [CarFormer](https://arxiv.org/pdf/2407.15843), [PlanT 2.0](https://arxiv.org/html/2511.07292v1), [CausalAgents](https://ar5iv.labs.arxiv.org/html/2207.03586)
- [Compositional Generalization from First Principles](https://proceedings.neurips.cc/paper_files/paper/2023/hash/15f6a10899f557ce53fe39939af6f930-Abstract.html), [Additive Decoders](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4ef594af0d9a519db8fb292452c461fa-Abstract.html), [HOWM](https://arxiv.org/pdf/2204.13661), [Are OC Representations Better at CG?](https://arxiv.org/html/2602.16689)
- [SAVi++](https://arxiv.org/pdf/2206.07764), [CODA](https://arxiv.deeppaper.ai/papers/2601.01224v1), [LPWM](https://arxiv.org/abs/2603.04553), [OC-STORM](https://arxiv.org/pdf/2501.16443)
- [DriveCombo](https://arxiv.org/html/2603.01637), [Fail2Drive](https://arxiv.org/html/2604.08535v1), [DA-WAM](https://arxiv.org/html/2608.19085), [Hydra-MDP](https://arxiv.org/abs/2406.06978), [DIPP](https://arxiv.org/abs/2207.10422), [DTPP](https://arxiv.org/abs/2310.05885), [PDM-Closed](https://arxiv.org/abs/2306.07962)
- Full text đã đọc: các PDF trong `ade2e-paper/` (danh mục `ade2e-paper/INDEX_HGO_2026-11-11.md`).
