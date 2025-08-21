# Anomaly-Detector
FIND-A 1차 프로젝트 알파팀 모듈 공유


1. 데이터 불러오기
2. IQR 기반 이상치 탐지
3. 1chart 변동성 스코어링

아직 그리드 써치하는 코드는 완성하지 못했고 코드 정리도 좀 덜 되어있기는 하지만 알고리즘 개선하실 때 통일된 함수로 하면 좋을 것 같아서 미리 공유드렸어요!!

functions.py에 필요한 함수들 있고 example.py에서 사용하는 예시 적어놓았습니다. 

example.py 돌리면 첨부파일에 같이 있는 png 파일대로 결과 나오면 맞는거고 마지막에 print하는 값은 0.0684620202397954
0.34212219357360485
이렇게 나오면 잘 돌아간것입니다!

쓴 라이브러리가 `pandas`, `numpy`, `mplfinance`밖에 없어서 버젼차이로 안돌아가고 할 건 없을 것 같은데 혹시 에러나면 알려주세요!!

-재연님-


>Score 공유(종완)

ETH
print(df_score["onechart_score"].mean()) : 0.03135716018707601
print(df_score[mask]["onechart_score"].mean()) : 0.44614434628215444

BERA
print(df_score["onechart_score"].mean() : 0.051387965329048414
print(df_score[mask]["onechart_score"].mean()) : 0.49410255477772713

YFI
print(df_score["onechart_score"].mean() : 0.0684620202397954
print(df_score[mask]["onechart_score"].mean()) : 0.34212219357360485
