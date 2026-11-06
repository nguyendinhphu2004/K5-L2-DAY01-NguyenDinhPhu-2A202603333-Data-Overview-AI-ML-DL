# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python: 3.13.16; PyTorch: 2.11.0+cu130; Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…

   class_id=468 ; class_name=cab ; rank=1 ; score=0.510915 ; taxonomy_name=ImageNet-1K

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

   Cả bức ảnh. Record chỉ có class_id, class_name, rank, score và kích thước ảnh (640×428). Nó không có bbox tọa độ, nên không trỏ tới vật nào cụ thể. Trong ảnh traffic có rất nhiều xe buýt, ô tô và người đi bộ, nhưng model chỉ trả về một danh sách top-5 nhãn cho toàn bộ khung hình.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

   Danh sách tên lớp đi kèm taxonomy và checkpoint. yolo11n-cls.pt được huấn luyện trên tập nhãn ImageNet-1K (1000 lớp), nên model chỉ có thể chọn trong 1000 tên đó. Model không tự nghĩ ra tên lớp mới.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

   Chỉ ghi tên lớp thì chưa đủ, vì:
   - class_id là chỉ số trong một taxonomy cụ thể. Cùng số 468 ở taxonomy khác sẽ là lớp khác.
   - class_name là chữ để người đọc hiểu. Tên có thể trùng nhau.
   - taxonomy_name cho biết class_id và class_name thuộc bảng nào (ImageNet-1K hay COCO-80).

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

   Với ảnh có nhiều vật hoặc nhiều chủ đề, guideline phải nói rõ tiêu chí chọn đúng một lớp: chọn theo vật chiếm diện tích lớn nhất, vật ở trung tâm, hay chủ đề tổng thể của cảnh. Guideline cũng phải nói rõ khi hai tiêu chí mâu thuẫn thì ưu tiên cái nào, và khi nào ảnh được coi là "không xác định / không thuộc taxonomy" thì gán nhãn đó hoặc escalate. Nếu không có quy tắc này, hai annotator sẽ chọn hai lớp khác nhau cho cùng một ảnh. Ví dụ ảnh traffic có thể là "bus" hoặc "car" tùy người.

6. Vì sao không được lấy `score` của model làm ground truth?

   score (0.510915) chỉ là mức "tin" của model, không được lấy làm ground truth vì:
   - Đây là prediction của máy, chưa qua người gán nhãn theo guideline.
   - Score cao không đảm bảo đúng. Ví dụ dining được gán restaurant với score 0.79, còn kitchen được gán gong với score 0.42 mà nhìn ảnh thì không phải cái chiêng.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…

   class_name=person ; score=0.912625 ; bbox_xyxy=[385.33, 69.24, 498.92, 348.92] ; bbox_width=113.58 ; bbox_height=279.68

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

   Box có hệ tọa độ xyxy, đơn vị pixel, gốc (0,0) ở góc trên-trái của ảnh, trục x sang phải, trục y xuống dưới.
   - Góc trên-trái: x=385.33, y=69.24, tức cách mép trái ảnh khoảng 385 px và cách mép trên khoảng 69 px.
   - Góc dưới-phải: x=498.92, y=348.92.
   - Rộng = 498.92 − 385.33 ≈ 113.58 px, cao = 348.92 − 69.24 = 279.68 px, đúng với bbox_width và bbox_height.
   - Trên ảnh detection_predictions.png, hộp xanh dương bao quanh người đứng quay lưng ở bên phải ảnh.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

   Hai ngưỡng so sánh trên ảnh kitchen là 0.35 và 0.60.
   - Ngưỡng 0.35: 11 prediction.
   - Ngưỡng 0.60: 6 prediction.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

   Hạ ngưỡng thì giữ lại thêm nhiều hộp có score thấp, nên reviewer phải kiểm tra thêm từng hộp mới:
   - hộp có thật sự bao một vật hay chỉ là nhiễu;
   - tên lớp có đúng không;
   - hộp có bao khít vật không, có trùng với hộp khác không;
   - có vật nào bị bỏ sót không.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

   Quy tắc box: Vẽ hộp chữ nhật sát khít quanh phần nhìn thấy được của vật, không chừa viền thừa và không cắt vào vật. Mỗi vật riêng biệt có đúng một hộp. Vật bị cắt ở mép ảnh thì hộp dừng ở mép ảnh.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

   Hộp person 0.61 ở góc dưới-trái ảnh kitchen (bbox [0.12, 263.18, 61.55, 310.78]) chỉ thấy một phần cánh tay, phần còn lại nằm ngoài khung hình, và chồng lên hộp oven 0.69 và bowl 0.50.
   - Việc vật bị cắt ở mép ảnh thì viết thành quy tắc trong guideline, vì tình huống này lặp lại ở nhiều ảnh.
   - Việc chỉ lộ một phần tay, không rõ có gán nhãn là người hay không, nếu guideline chưa nói thì đưa lên Lab Coach, không tự quyết, vì mỗi annotator tự chọn thì dữ liệu sẽ không nhất quán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…

   instance_id=kitchen-001 ; class_name=person ; score=0.899318 ; polygon_point_count=348 ; 5 điểm đầu của polygon_xy = [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

   Box chỉ cho biết vật nằm trong một hình chữ nhật. Polygon cho biết pixel nào thật sự thuộc vật:
   - Hộp của người rộng 113 px, cao 280 px, chứa cả nền tường và nồi chảo. Polygon 348 điểm ôm sát đầu, vai, hai tay chống hông và hai chân, bỏ phần nền ra ngoài (xem segmentation_prediction.png).
   - Từ polygon tính được hình dạng, diện tích thật và biên của vật, điều mà box không làm được.
   - Đổi lại số điểm nhiều hơn (348 điểm so với 4 tọa độ của box), nên công gán nhãn và QC cũng nhiều hơn.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

   instance_id (ví dụ kitchen-001) chỉ là số thứ tự để phân biệt từng cá thể trong cùng một ảnh.
   - Không phải class ID: class ID (class_id=0 là person) cho biết vật thuộc loại gì, còn instance_id chỉ cho biết đây là vật thứ mấy. Hai bowl khác nhau có cùng class_id nhưng instance_id khác nhau.
   - Không phải tracking ID: tracking ID dùng để theo một vật qua nhiều khung hình video. instance_id chỉ có nghĩa trong một ảnh, sang ảnh khác thì đánh số lại từ đầu.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

   Quy tắc vẽ biên mask: Vẽ biên polygon men theo đường viền ngoài của vật ở mức pixel, chỉ bao phần nhìn thấy được của đúng vật đó, không lấn sang nền hay vật khác. Mỗi vật có một mask riêng, không gộp các vật chạm nhau vào chung một mask.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

   Mask dining table 0.58 (kitchen-005) trùm lên cả bàn và các vật đặt trên bàn như bát, hũ, nên biên giữa bàn và vật trên bàn rất mờ. Mask oven 0.49 (kitchen-007) cũng chồng lên mask bàn ở góc dưới-trái.
   - Dùng guideline cho việc chung: vật đặt trên bàn được vẽ mask riêng, mask bàn bỏ phần bị vật khác che, vùng hai vật chồng nhau thuộc vật nằm phía trước.
   - Dùng escalation cho ca guideline chưa nói tới, ví dụ vùng tối lẫn giữa lò và bàn mà không phân biệt được vật nào, hay mảnh cánh tay ở mép ảnh (kitchen-009) có được coi là instance person hay không.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)                               | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                         | Annotator làm gì?                                                                                              | Reviewer xem gì?                                                                                                                   |
| --------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | Một nhãn cho cả ảnh, lấy từ taxonomy (class_id, class_name, taxonomy_name). | Ảnh traffic nhiều xe nhưng model chỉ trả cab (0.51), ảnh kitchen bị gán gong (0.42): nhãn sai hoặc không có lớp phù hợp trong ImageNet-1K. | Đọc guideline, chọn một lớp theo tiêu chí ưu tiên. Gặp ảnh mơ hồ hoặc ngoài taxonomy thì escalate, không đoán. | Đối chiếu nhãn với ảnh và guideline, không dựa vào score. Trả lại ca sai hoặc không nhất quán.                                     |
| Phát hiện vật thể     | Một hộp xyxy (pixel) và một lớp cho mỗi vật.                                | Ảnh kitchen: person 0.61 chỉ thấy tay, cắt ở mép ảnh và chồng lên oven 0.69. Hạ ngưỡng từ 0.60 xuống 0.35 thì từ 6 lên 11 hộp.             | Vẽ hộp khít quanh phần nhìn thấy, mỗi vật một hộp. Vật bị che hoặc cắt mà guideline chưa có thì escalate.      | Kiểm tra hộp có khít không, có thiếu hoặc thừa hộp không, lớp có đúng không. Duyệt kỹ các hộp score thấp.                          |
| Instance segmentation | Một polygon mask cho mỗi vật, kèm lớp và instance_id.                       | Ảnh kitchen: mask dining table 0.58 bao cả vật đặt trên bàn, mask oven 0.49 chồng lên mask bàn, biên mờ.                                   | Vẽ biên men theo viền vật, tách riêng từng vật, bỏ phần bị vật khác che. Ca khó thì escalate.                  | Soi biên polygon có khít không, có lẫn nền hay vật bên cạnh không, các vật chạm nhau có bị gộp không. Yêu cầu vẽ lại nếu biên sai. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

### BÀI LÀM

1. Chỉ dùng các ảnh công khai COCO 2017 đã cung cấp (có ghi nguồn trong IMAGE_ATTRIBUTION.md), không đưa họ tên, MSSV, email, số điện thoại hay dữ liệu cá nhân vào báo cáo và output.

2. Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi, tôi sẽ dừng lại, không xử lý hay tải lên, và báo ngay cho Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
