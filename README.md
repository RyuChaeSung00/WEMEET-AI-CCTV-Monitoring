삐뽀(PPIBBO) - 지능형 실시간 AI 모바일 관제 서비스
대학가 치안 사각지대 및 생활 불법행위 해소를 위한 YOLOv8 기반 통합 관제 어플리케이션

1. 프로젝트 개요
대학가 및 원룸촌 등 관리 사각지대에서 발생하는 흡연, 무단 투기, 전동 킥보드 위반 등 생활형 불법행위를 AI로 실시간 감지하여 관리자에게 알림을 제공하는 시스템입니다. 기존 고정형 모니터링의 한계를 극복하고 관제 업무의 이동성과 실시간성을 확보하는 데 집중했습니다.

2. 핵심 기능
AI 실시간 행동 감지: YOLOv8-Pose를 활용해 코와 손목 사이의 유클리드 거리를 측정하여 흡연 동작을 판별하고, 물체와의 벡터 거리 변화를 추적하여 무단 투기를 감지합니다.
<img width="530" height="964" alt="image" src="https://github.com/user-attachments/assets/641a4103-b069-4a46-8442-ba9b511ae83f" />

프라이버시 보호 기능: 행인의 초상권 침해 방지를 위해 평상시 모니터링 화면에서는 얼굴 영역을 실시간으로 블러 처리하여 송출합니다.
<img width="1404" height="636" alt="image" src="https://github.com/user-attachments/assets/78d9023d-9c4c-4056-91c0-405ae2409ee2" />

스마트 알림 및 경고: 위반 상황이 감지되면 화면 테두리가 붉은색으로 점멸하며 시각적 경고를 보내고, 동시에 현장에 TTS 안내 방송을 자동으로 송출합니다.
<img width="582" height="972" alt="image" src="https://github.com/user-attachments/assets/e4d437b2-8d5a-44e6-8317-97c68b6b85a0" />
<img width="586" height="976" alt="image" src="https://github.com/user-attachments/assets/817761a7-52e8-4c72-9d3e-5f8edf27c2fa" />
<img width="582" height="1018" alt="image" src="https://github.com/user-attachments/assets/7237875b-5021-4ade-8343-b77b5af7ac72" />


실시간 통계 대시보드: 위반 유형별 발생 빈도를 도넛 차트로 시각화하여 관리자가 주요 이슈를 한눈에 파악할 수 있도록 지원합니다.
<img width="447" height="735" alt="image" src="https://github.com/user-attachments/assets/2b479308-b6f4-46c3-9777-87a39a76f524" />


3. 기술 스택
AI 및 비전: Python, YOLOv8 (Pose/Detect), OpenCV

백엔드: FastAPI, ngrok (네트워크 터널링)

프론트엔드: HTML, CSS, JavaScript (Hybrid App), Chart.js

기타 도구: pyttsx3 (TTS)

4. 성과 및 성장 포인트
주도적 역할 수행: 작년 프로젝트에서는 팀원으로서 주어진 모듈 구현에 집중했으나, 이번 프로젝트에서는 기술 설계부터 시스템 통합까지 전 과정을 주도적으로 이끌며 리더십을 발휘했습니다.

실전 문제 해결: 로컬 서버의 외부 접속 한계를 ngrok 터널링 기술로 해결하고, 개인정보 보호를 위한 조건부 블러 처리 로직을 직접 고안하여 구현했습니다.

시스템 최적화: 지속적인 테스트와 튜닝을 통해 위반 행위 감지 정확도를 97%에서 99% 수준까지 확보하여 신뢰도 높은 관제 환경을 구축했습니다.
