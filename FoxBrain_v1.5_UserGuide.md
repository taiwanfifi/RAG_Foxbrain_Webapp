# FoxBrain 1.5 模型評測說明

## 一、概述

本文件提供 FoxBrain 1.5 模型之評測指引，供評測人員透過 API 進行模型能力與表現評估。

---

## 二、服務資訊

| 項目 | 說明 |
|------|------|
| **服務位址** | `http://4.151.237.144:8000` |
| **模型名稱 (Model ID)** | `20251203_remove_repeat` |
| **API 金鑰** | `token-abc123` |
| **最大上下文長度** | 32,768 tokens |
| **API 相容性** | OpenAI 相容格式 (v1) |

---

## 三、認證方式

所有 API 請求需在 HTTP Header 中帶入 API 金鑰：

```
Authorization: Bearer token-abc123
```

未帶入或金鑰錯誤將回傳 `401 Unauthorized`。

---

## 四、評測方式

### 4.1 查詢可用模型

```bash
curl -s -H "Authorization: Bearer token-abc123" \
  "http://4.151.237.144:8000/v1/models"
```

### 4.2 文字補全 / 對話 (Chat Completions)

**請求範例：**

```bash
curl -s -X POST "http://4.151.237.144:8000/v1/chat/completions" \
  -H "Authorization: Bearer token-abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "20251203_remove_repeat",
    "messages": [
      {"role": "system", "content": "你是一個專業又簡潔的技術助理。"},
      {"role": "user", "content": "什麼是 vLLM？"},
      {"role": "assistant", "content": "vLLM 是一個高效能的 LLM 推論引擎。"},
      {"role": "user", "content": "它為什麼比一般 PyTorch 推論快？"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

**Python 範例：**

```python
import requests

url = "http://4.151.237.144:8000/v1/chat/completions"
headers = {
    "Authorization": "Bearer token-abc123",
    "Content-Type": "application/json"
}
payload = {
    "model": "20251203_remove_repeat",
    "messages": [
      {"role": "system", "content": "你是一個專業又簡潔的技術助理。"},
      {"role": "user", "content": "什麼是 vLLM？"},
      {"role": "assistant", "content": "vLLM 是一個高效能的 LLM 推論引擎。"},
      {"role": "user", "content": "它為什麼比一般 PyTorch 推論快？"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

### 4.3 單輪補全 (Completions)

```bash
curl -s -X POST "http://4.151.237.144:8000/v1/completions" \
  -H "Authorization: Bearer token-abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "20251203_remove_repeat",
    "prompt": "人工智慧的定義是：",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

---

## 五、常用參數說明

| 參數 | 說明 | 建議範圍 |
|------|------|----------|
| `model` | 必填，固定為 `20251203_remove_repeat` | — |
| `messages` | 對話內容 (chat 介面) | 陣列，每則含 `role`、`content` |
| `prompt` | 單一輸入文字 (completions 介面) | 字串 |
| `max_tokens` | 生成之最大 token 數 | 依題目需求，勿超過 32768 |
| `temperature` | 隨機度，愈高愈隨機 | 0.0～2.0，評測可多用 0.7 |
| `top_p` | nucleus sampling 門檻 | 0.0～1.0 |
| `stream` | 是否串流輸出 | `true` / `false` |

---

## 六、評測建議項目

1. **一般問答**：事實、常識、簡短推理。
2. **指令遵循**：格式、長度、角色設定是否遵守。
3. **長文與上下文**：多輪對話、長 context 理解與摘要。
4. **安全性與邊界**：敏感請求、越獄嘗試之回應是否合理。
5. **穩定性**：連續請求、併發、長時間評測是否穩定。

---

## 七、注意事項

- 請妥善保管 API 金鑰，勿公開於版本控制或公開文件。
- 單次輸入＋輸出的總 token 數請勿超過 **32,768**。
- 若遇連線逾時，可檢查網路與服務狀態；服務版本為 vLLM 0.10.2。

---

## 八、聯絡與回饋

評測過程若發現異常、錯誤或建議，請跟窗口回報，以利模型改進。
