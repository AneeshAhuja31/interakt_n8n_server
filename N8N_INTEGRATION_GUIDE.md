# n8n Integration Guide - Quick Reference

## What You Need to Change in n8n

### Step-by-Step Changes

#### 1. Delete Code1 Node
- In your n8n workflow, find and **delete** the "Code1" node
- This is the JavaScript node that currently does intent classification

#### 2. Add HTTP Request Node

Add a new **HTTP Request** node where Code1 was.

**Connection:**
```
[Return Records + Webhook Payload] → [HTTP Request] → [Intent]
```

#### 3. Configure HTTP Request Node

Open the HTTP Request node and configure:

**Basic Auth/Headers Tab:**
- No authentication needed
- Add header: `Content-Type: application/json`

**Parameters Tab:**

| Field | Value |
|-------|-------|
| **Method** | POST |
| **URL** | `http://localhost:8000/classify-intent` (or your deployed URL) |
| **Send Body** | ✅ Yes |
| **Body Content Type** | JSON |
| **Specify Body** | Using JSON |

**JSON Body** (copy this exactly):

```json
{
  "contact_record": {
    "phone_number": "={{ $json.contact_record.phone_number }}",
    "active_template_flow": "={{ $json.contact_record.active_template_flow }}",
    "last_message": "={{ $json.contact_record.last_message }}"
  },
  "webhook_payload": {
    "body": {
      "data": {
        "customer": {
          "phone_number": "={{ $json.webhook_payload.body.data.customer.phone_number }}"
        },
        "message": {
          "message_content_type": "={{ $json.webhook_payload.body.data.message.message_content_type }}",
          "message": "={{ $json.webhook_payload.body.data.message.message }}",
          "button_text": "={{ $json.webhook_payload.body.data.message.button_text }}",
          "button_payload": "={{ $json.webhook_payload.body.data.message.button_payload }}",
          "message_context": "={{ $json.webhook_payload.body.data.message.message_context }}"
        }
      }
    }
  }
}
```

**Options Tab:**
- **Response Format**: JSON
- **Timeout**: 30000 (30 seconds)

#### 4. Rename Node (Optional)

Rename the HTTP Request node to:
- "AI Intent Classifier" or
- "Code1 Replacement"

#### 5. No Other Changes Needed!

The **Intent** (Switch) node and all other nodes remain **exactly the same**.

The API returns the same output structure as Code1, so everything downstream works without changes.

---

## Expected Output

The API will return (same as Code1):

```json
{
  "phone_number": "+1234567890",
  "user_message": "Hi, is this Star Trade?",
  "message_type": "text",
  "detected_intent": "welcome_greeting",
  "active_template_flow": "none",
  "last_message": "",
  "full_context": { ... },
  "check_if_quick_message": false
}
```

---

## Testing

### 1. Start the API

Windows:
```bash
start.bat
```

Linux/Mac:
```bash
bash start.sh
```

### 2. Verify API is Running

Open browser: http://localhost:8000

You should see:
```json
{
  "status": "running",
  "service": "WhatsApp Intent Classifier",
  "version": "1.0.0",
  "model": "gemini-1.5-flash"
}
```

### 3. Test in n8n

1. Activate your workflow
2. Send a test WhatsApp message
3. Check the HTTP Request node output
4. Verify the Intent Switch routes correctly

---

## Troubleshooting

### API Not Reachable

**Error in n8n**: Connection refused

**Solutions**:
1. Check API is running: `curl http://localhost:8000`
2. If using WSL/Docker, use correct IP instead of localhost
3. Check Windows Firewall settings

### Wrong Output Format

**Error**: Intent Switch doesn't route correctly

**Solution**:
1. Check HTTP Request node output in n8n
2. Verify Response Format is set to "JSON"
3. Check the API logs for errors

### Slow Response

**Issue**: HTTP Request times out

**Solutions**:
1. Increase timeout to 60000 (1 minute)
2. Check your internet connection (Gemini API requires internet)
3. Consider upgrading to faster Gemini model or adding caching

---

## Production Deployment

For production, deploy the API to:

- **Railway** (easiest): Connect GitHub, auto-deploy
- **Render**: Free tier available
- **Your VPS**: Use gunicorn or Docker

Then update the URL in n8n:
```
http://your-domain.com/classify-intent
```

---

## Visual Reference

### Before (Code1 Node)
![Before]
```
┌─────────────────────────────────┐
│ Return Records + Webhook Payload│
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ Code1 (JavaScript)              │
│ - Regex pattern matching        │
│ - Hardcoded intent rules        │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ Intent (Switch)                 │
└─────────────────────────────────┘
```

### After (HTTP Request Node)
```
┌─────────────────────────────────┐
│ Return Records + Webhook Payload│
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ HTTP Request                    │
│ - POST to AI API                │
│ - Gemini-powered classification │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ Intent (Switch)                 │
└─────────────────────────────────┘
```

---

**That's it! Your n8n workflow now uses AI-powered intent classification!** 🚀
