# 🤖 Llama-KoEmpathy

Llama-KoEmpathy는 LLaMA 기반의 한국어 감정 인식 챗봇 모델입니다. 해당 레포에서는 학습 코드를 공유하고 있습니다.<br>
**해당 프로젝트는 단대소고 포트폴리오 제출을 위해 제작되었습니다.**

*Built with Llama*


## ✨ Features
- 간편한 데이터 전처리 및 토크나이징
- LoRA 기반의 효율적인 학습 파이프라인
- 학습 과정 모니터링 및 로깅
- 손쉬운 모델 평가 및 추론

## ⚠️ License Requirements
이 프로젝트를 사용할 때 다음 사항을 준수해야 합니다:
- Meta의 [Acceptable Use Policy](https://llama.meta.com/llama3_1/use-policy)를 따라야 합니다
- 월간 활성 사용자가 7억명을 초과하는 제품/서비스에 사용할 경우 Meta의 별도 라이선스가 필요합니다
- 
  
## ⚙️ Installation
```bash
git clone https://github.com/byeolki/Llama-KoEmpathy.git
cd Llama-KoEmpathy
pip install -r requirements.txt
```

## 🚀 Usage
자세한 사용법은 `tutorial.ipynb`를 참고해주세요.

학습된 모델을 사용하시려면 [HuggingFace Hub](https://huggingface.co/byeolki/Llama-KoEmpathy)를 방문해주세요.

## 🙏 Credits & References
이 프로젝트는 다음 오픈소스 프로젝트들을 참고 및 활용했습니다:

- [Llama 3.1](https://llama.meta.com/) - Meta AI의 Language Model (Llama 3.1 Community License)
  - 공감 능력을 가진 한국어 챗봇을 위해 파인튜닝한 [Llama-KoEmpathy](https://huggingface.co/byeolki/Llama-KoEmpathy) 모델을 사용합니다

## ⚖️ License
이 프로젝트의 코드는 MIT License에 따라 배포됩니다:

MIT License

Copyright (c) 2024 Byeolki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

단, 이 프로젝트에 포함된 외부 라이브러리들은 각각의 라이센스를 따릅니다:
- Llama 모델 및 관련 코드: Llama 3.1 Community License

## 📕 Notice
Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
