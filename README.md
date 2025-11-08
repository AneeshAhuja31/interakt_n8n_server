# WhatsApp Intent Classifier API

AI-powered intent classification for WhatsApp customer service using **FastAPI**, **LangChain**, and **Google Gemini**.

This service replaces the hardcoded regex-based Code1 node in your n8n workflow with intelligent, context-aware intent detection.

## Features

- **AI-Powered Classification**: Uses Google Gemini (gemini-1.5-flash) for intelligent intent detection
- **Structured Output**: Pydantic models with Literal types ensure type-safe, validated responses
- **Context-Aware**: Considers conversation history and active flows
- **Multi-language Support**: Handles English and Hindi/Hinglish messages
- **Drop-in Replacement**: Matches Code1 node input/output structure exactly

## Intent Categories

The API classifies messages into these intents:

1. `welcome_greeting` - Greetings and conversation starters
2. `availability_check_with_link` - Product availability, pricing inquiries
3. `order_confirmation_simple` - Order placement requests
4. `order_address_collect` - Delivery address collection
5. `order_payment_choice` - Payment method selection
6. `order_payment_confirmed` - Payment completion confirmations
7. `alternate_suggestion_simple` - Alternative product requests
8. `out_of_stock_simple` - Out of stock discussions
9. `product_availability_query` - Product-specific queries in availability flow
10. `order_confirmation_approval` - Order approval in confirmation flow
11. `general_text_message` - Other text messages
12. `unknown` - Unclassifiable messages

## Installation & Setup

### 1. Prerequisites

- Python 3.9 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### 2. Install Dependencies

```bash
cd intent_classifier_api
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

### 4. Run the Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### 5. Test the API

Open your browser and visit:
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/

Quick test using curl:

```bash
curl -X POST "http://localhost:8000/test-classification" \
  -H "Content-Type: application/json" \
  -d '{"user_message": "Hi, is this Star Trade?", "active_flow": "none"}'
```

## n8n Integration

### Replace Code1 Node with HTTP Request Node

Follow these steps to integrate the API into your n8n workflow:

#### Step 1: Deploy the API

Make sure the FastAPI server is running and accessible to n8n. Options:

- **Local Development**: Run on `http://localhost:8000`
- **Production**: Deploy to a cloud service (Railway, Render, Fly.io, etc.)
- **Internal Server**: Run on your internal server

#### Step 2: Remove Code1 Node

1. Open your n8n workflow
2. **Delete** the "Code1" node (the JavaScript code node)

#### Step 3: Add HTTP Request Node

1. Add a new **HTTP Request** node in place of Code1
2. Connect it to the **"Return Records + Webhook Payload"** (Postgres) node

#### Step 4: Configure HTTP Request Node

**Basic Settings:**
- **Method**: `POST`
- **URL**: `http://your-server:8000/classify-intent`
  - Replace `your-server` with:
    - `localhost` (if running locally)
    - Your server IP/domain (if deployed)

**Headers:**
```json
{
  "Content-Type": "application/json"
}
```

**Body:**
- **Body Content Type**: `JSON`
- **Specify Body**: `Using JSON`
- **JSON Body**:

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

**Options:**
- **Response Format**: `JSON`
- **Timeout**: `30000` (30 seconds)

#### Step 5: Rename the Node (Optional)

Rename the HTTP Request node to **"AI Intent Classifier"** or **"Code1 Replacement"** for clarity.

#### Step 6: Connect to Intent Switch Node

Connect the output of the HTTP Request node to the **"Intent"** (Switch) node, which routes based on `detected_intent`.

#### Step 7: Test the Workflow

1. Send a test message to your WhatsApp webhook
2. Check the HTTP Request node output - it should have the same structure as Code1:

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

### Visual n8n Workflow Change

**Before:**
```
[Return Records + Webhook Payload] → [Code1 (JavaScript)] → [Intent (Switch)]
```

**After:**
```
[Return Records + Webhook Payload] → [HTTP Request: AI Intent Classifier] → [Intent (Switch)]
```

## API Endpoints

### POST /classify-intent

Main endpoint for intent classification.

**Request Body:**
```json
{
  "contact_record": {
    "phone_number": "string",
    "active_template_flow": "string",
    "last_message": "string"
  },
  "webhook_payload": {
    "body": {
      "data": {
        "customer": {
          "phone_number": "string"
        },
        "message": {
          "message_content_type": "string",
          "message": "string",
          "button_text": "string",
          "button_payload": "string",
          "message_context": {}
        }
      }
    }
  }
}
```

**Response:**
```json
{
  "phone_number": "string",
  "user_message": "string",
  "message_type": "string",
  "detected_intent": "welcome_greeting",
  "active_template_flow": "string",
  "last_message": "string",
  "full_context": {},
  "check_if_quick_message": false
}
```

### POST /test-classification

Quick test endpoint for development.

**Query Parameters:**
- `user_message` (required): The message to classify
- `active_flow` (optional): Current conversation flow (default: "none")
- `last_message` (optional): Previous message context

**Response:**
```json
{
  "detected_intent": "availability_check_with_link",
  "confidence": "high",
  "reasoning": "User is asking about product availability"
}
```

### GET /

Health check endpoint.

**Response:**
```json
{
  "status": "running",
  "service": "WhatsApp Intent Classifier",
  "version": "1.0.0",
  "model": "gemini-1.5-flash"
}
```

## Deployment Options

### Option 1: Local Development

```bash
python main.py
```

Access at: `http://localhost:8000`

### Option 2: Production with Gunicorn

```bash
pip install gunicorn
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Option 3: Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t intent-classifier .
docker run -p 8000:8000 --env-file .env intent-classifier
```

### Option 4: Cloud Platforms

Deploy to:
- **Railway**: Connect GitHub repo, auto-deploy
- **Render**: Web service from GitHub
- **Fly.io**: `fly launch`
- **Google Cloud Run**: Serverless container deployment

## Advantages Over Code1 Regex Approach

| Feature | Code1 (Regex) | AI Classifier |
|---------|--------------|---------------|
| **Flexibility** | Fixed patterns only | Understands context and variations |
| **Maintenance** | Manual regex updates | Self-adapting to language patterns |
| **Language Support** | Hardcoded patterns | Natural language understanding |
| **New Intents** | Code changes required | Just update prompt instructions |
| **Typos & Variations** | Often fails | Handles naturally |
| **Context Awareness** | Basic rule-based | Deep contextual understanding |

## Troubleshooting

### API Key Issues

**Error**: `GOOGLE_API_KEY not found in environment variables`

**Solution**: Ensure `.env` file exists with valid `GOOGLE_API_KEY`

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'langchain_google_genai'`

**Solution**: Reinstall dependencies:
```bash
pip install -r requirements.txt
```

### n8n Connection Issues

**Error**: HTTP Request node shows connection error

**Solution**:
1. Verify API is running: `curl http://localhost:8000`
2. Check firewall/network settings
3. Ensure correct URL in n8n HTTP Request node

### Classification Accuracy Issues

**Problem**: Intents are misclassified

**Solution**:
1. Check the prompt in [main.py](main.py:107-162) - you can refine intent descriptions
2. Adjust temperature (currently 0.1) in [main.py](main.py:80) for more/less variation
3. Add more examples in the prompt template

## Customization

### Add New Intents

1. Update the `Literal` type in [main.py](main.py:33-49):
```python
detected_intent: Literal[
    "welcome_greeting",
    "your_new_intent",  # Add here
    ...
]
```

2. Add intent description in the prompt template ([main.py](main.py:107-162))

3. Update your n8n **Intent** (Switch) node to handle the new intent

### Change AI Model

In [main.py](main.py:77-84), change:
```python
model="gemini-1.5-pro"  # More capable but slower
# or
model="gemini-1.5-flash"  # Faster, good balance (default)
```

### Adjust Temperature

In [main.py](main.py:80):
```python
temperature=0.1  # Very consistent (0.0-0.2)
temperature=0.5  # More creative (0.3-0.7)
```

## License

MIT

## Support

For issues or questions:
1. Check the [FastAPI docs](http://localhost:8000/docs) when server is running
2. Review logs for error details
3. Test using `/test-classification` endpoint first

---

**Built with**: FastAPI • LangChain • Google Gemini • Pydantic
