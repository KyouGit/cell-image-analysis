# Sample Images — BCCD Dataset

`examples/demo.py` 실행용 샘플 이미지 5장.

출처: [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset) (MIT License)  
3 classes: WBC / RBC / Platelet

## 사용법

```bash
# 랜덤 샘플 자동 선택
python examples/demo.py

# 특정 이미지 지정
python examples/demo.py --image data/sample/BloodImage_00001.jpg
python examples/demo.py --image data/sample/BloodImage_00001.jpg --save result.jpg
```
