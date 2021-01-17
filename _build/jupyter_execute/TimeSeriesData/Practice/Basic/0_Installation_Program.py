#!/usr/bin/env python
# coding: utf-8

# # **0. 컴퓨터와의 소통을 위한 도구 소개**  
# 
# > **Python**
# > - 간결하고 쉬운 컴퓨터와의 소통언어
# > - 데이터분석과 머신러닝을 위한 수많은 라이브러리를 포함
# > - 다양한 확장이 용이(ex. R, SPSS, etc.)
# 
# > **Anaconda**
# > - Python기반의 Open Data Science Platform
# > - Python을 포함하여 Python Library 등을 하나로 정리해 둔 배포판
# > - Pandas, Numpy, Matplotlib 등 데이터분석에 유용한 Library를 포함
# > - 추가적인 Library를 연결 및 확장하여 개발/웹/오피스/비즈니스로 활용가능
# >> - 어떤 라이브러리를 써야할지 잘 모르겠을때: https://github.com/vinta/awesome-python
# >> - 주목받는 파이썬 프로젝트들을 둘러보고 싶을때: https://github.com/trending/python
# 
# > **Jupyter Notebook / Jupyter Lab**
# > - Interactive 환경을 인간에게 제공하여 컴퓨터(Programming Platform)와 소통을 쉽게함
# > - Anaconda 프로그래밍 환경을 그대로 사용
# > - 코딩하면서 바로 결과를 확인할 수 있음
# > - 문서화(Markdown)와 클라우드(Git, Google Drive 등) 연결/저장 등이 가능
# > - 각종 단축키와 튜터리얼 등의 자료가 많음

# # **1. 분석환경 세팅하기(윈도우10기준)**  

# ## **1-1) 기본세팅: Anaconda & Jupyter**
# 
# **1) PC 내 제어판: 이전에 설치된 기록이 있다면 삭제**  
# -> 시작 -> "제어판" 타이핑 -> "제어판" 선택  
# -> 프로그램및기능 선택  
# -> Anaconda3 또는 Python으로 시작되는 이름 선택  
# -> 우클릭으로 제거  
# **2) 인터넷: 프로그램 다운로드**  
# -> https://www.anaconda.com/download/ 접속  
# -> Windows/macOX/Linux중 Windows 클릭  
# -> Python "Download" 클릭  
# **3) PC: 다운로드한 프로그램 실행 및 설치**  
# -> 다운로드한 설치파일 우클릭으로 "관리자 권한으로 실행"  
# ->"Next"를 누르며 진행하면 되는데 설치 중 "Advanced Options" 페이지에서  
# "Add Anaconda to the system PATH environment variable"을 선택을 체크하고 진행  
# (본 링크 https://hobbang143.blog.me/221461726444 "설치 02" 부터 동일하게 참조)  

# ## **1-2) 고급세팅: PIP & Jupyter Notebook & Jupyter Lab 업데이트 및 확장**
# 
# **1) PC: Anaconda Prompt 접속**  
# -> 시작 -> "Anaconda" 타이핑  
# -> "Anaconda Prompt" 우클릭으로 "관리자 권한으로 실행"  
# **2) Anaconda Prompt: PIP & Jupyter Notebook & Jupyter Lab 업데이트 및 확장 한번에 설치**  
# **(하기 내용 복사 후 Anaconda Prompt에 우클릭(붙여넣기))**  
# :: Update of PIP  
# pip install --upgrade pip  
# python -m pip install --user --upgrade pip  
# :: Jupyter Nbextensions  
# pip install jupyter_contrib_nbextensions  
# jupyter contrib nbextension install --user  
# :: Jupyter Lab  
# pip install jupyterlab  
# pip install --upgrade jupyterlab  
# :: Jupyter Lab Extensions Package  
# pip install nodejs  
# conda install --yes nodejs  
# conda install -c conda-forge --yes nodejs  
# :: Table of Contents  
# jupyter labextension install @jupyterlab/toc    
# :: Shortcut UI  
# jupyter labextension install @jupyterlab/shortcutui  
# :: Variable Inspector  
# jupyter labextension install @lckr/jupyterlab_variableinspector  
# :: Go to Definition of Module  
# jupyter labextension install @krassowski/jupyterlab_go_to_definition    
# :: Interactive Visualization  
# jupyter labextension install @jupyter-widgets/jupyterlab-manager    
# jupyter labextension install lineup_widget  
# :: Connection to Github  
# jupyter labextension install @jupyterlab/github   
# :: CPU+RAM Monitor  
# pip install nbresuse    
# jupyter labextension install jupyterlab-topbar-extension jupyterlab-system-monitor    
# :: File Tree Viewer  
# jupyter labextension install jupyterlab_filetree  
# :: Download Folder as Zip File  
# conda install --yes jupyter-archive  
# jupyter lab build  
# jupyter labextension update --all  
# :: End  
#     

# ## **1-3) 선택세팅: Jupyter Lab에 R연결**
# 
# **1) 인터넷: R프로그램 다운로드 및 설치** 
# -> https://cran.r-project.org/bin/windows/base/ 접속  
# -> "Download" 클릭 및 실행  
# **2) 인터넷: Git 설치**  
# -> https://git-scm.com 접속  
# -> "Download" 클릭 및 실행  
# **3) PC: Anaconda Prompt 접속**  
# -> 시작 -> "Anaconda" 타이핑  
# -> "Anaconda Prompt" 우클릭으로 "관리자 권한으로 실행"  
# **4) Anaconda Prompt: Jupyter Client & R설치**  
# -> conda install --yes -c anaconda jupyter_client  
# -> conda install --yes -c r r-essentials  
# **5) Anaconda Prompt: R패키지 설치**
# -> cd C:\Program Files\R\R-3.6.1\bin(R이 설치된 경로로 이동)
# -> R.exe   
# -> install.packages("devtools")  
# -> devtools::install_github("IRkernel/IRkernel")  
# -> IRkernel::installspec(user = FALSE) 

# # **2. 분석준비 설정 및 강의시작(윈도우10기준)**  

# ## **2-1) Jupyter Notebook & Jupyter Lab 열기 및 내부 설정**
# 
# **1) PC: Jupyter Notebook 속성 진입**  
# -> 시작 -> "Jupyter" 타이핑  
# -> "Jupyter Notebook" 우클릭으로 "작업 표시줄에 고정"  
# -> 작업 표시줄의 "Jupyter Notebook" 아이콘 우클릭  
# -> 상단 "Jupyter Notebook" 우클릭 -> 속성 클릭  
# **2) PC: Jupyter Notebook 작업경로 반영**  
# -> "대상"에서 "%USERPROFILE%/" 삭제후 본인 작업 폴더 반영(ex. D:\)  
# -> (필요시:기존) jupyter-notebook-script.py (필요시:변경) jupyter-lab-script.py  
# -> "시작위치"에서 %HOMEPATH% 삭제후 본인 작업 폴더 반영(ex. D:\)  
# -> 하단 "확인" 클릭  
# **3) PC: Jupyter Notebook 실행**  
# -> 작업 표시줄의 "Jupyter Notebook" 좌클릭  
# -> (만약 인터넷 창이 반응이 없다면) 인터넷 주소창에 http://localhost:8888/tree 타이핑  
# **4) PC: Jupyter Notebook Nbextensions 기능 추가**  
# -> Files/Running/Clusters/Nbextensions 중 Nbextensions 클릭  
# -> "disable configuration for nbextensions without explicit compatibility" 체크 해제  
# -> 하기 7종 클릭하여 기능확인 후 필요시 추가(Table of Contents는 수업용 필수)  
# - Table of Contents
# - Autopep8
# - Codefolding
# - Collapsible Headings
# - Hide Input All
# - Execute Time
# - Variable Inspector
# 
# **5) PC: Jupyter Notebook으로 강의시작(이론대응)**  
# -> Files/Running/Clusters/Nbextensions 중 Files 클릭  
# -> 다운로드 받은 강의자료의 폴더위치로 이동하여 자료 실행  
# **6) PC: Jupyter Lab으로 강의시작(실습대응): 확장기능설치는 1-2)에 이미 포함됨**  
# -> 인터넷 주소창에서 http://localhost:8888/lab 타이핑  
# -> 좌측 상단 폴더에서 다운로드 받은 강의자료의 폴더위치로 이동하여 자료 실행  

# # **별첨. 이상 무!**
# 
# - 위 내용들이 순서대로 진행되야 하며, 제대로 진행하셨다면 문제 발생 확률 낮음  
# - 만약 이상이 있다면 모두 삭제 후 처음부터 하나씩 다시 진행하시길 권장  
