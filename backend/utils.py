
import os
import logging
import uuid
from datetime import datetime, timedelta
from typing import List
# from transformers import pipeline, MarianMTModel, MarianTokenizer  # Import on demand
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Global cache for translators
translator_cache = {}

def get_translator(source_lang: str, target_lang: str):
	"""Get translator model from cache or load it"""
	try:
		from transformers import MarianMTModel, MarianTokenizer
	except ImportError:
		return None

	# Local definition to avoid import issues
	HELSINKI_MODELS = {
		"en-es": "Helsinki-NLP/opus-mt-en-es",
		"es-en": "Helsinki-NLP/opus-mt-es-en",
	}

	model_key = f"{source_lang}-{target_lang}"
	if model_key not in translator_cache:
		if model_key not in HELSINKI_MODELS:
			return None
		try:
			model_name = HELSINKI_MODELS[model_key]
			tokenizer = MarianTokenizer.from_pretrained(model_name)
			model = MarianMTModel.from_pretrained(model_name)
			from transformers import pipeline
			translator_cache[model_key] = pipeline("translation", model=model, tokenizer=tokenizer)
		except Exception as e:
			logger.error(f"Failed to load translator {model_key}: {e}")
			return None
	return translator_cache[model_key]

def translate_with_chunking(translator, text: str, max_chunk_size: int = 512) -> str:
	"""Translate text by splitting into chunks to avoid token limits"""
	sentences = text.split('. ')
	chunks = []
	current_chunk = ""
	for sentence in sentences:
		if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
			chunks.append(current_chunk)
			current_chunk = sentence
		else:
			current_chunk += ". " + sentence if current_chunk else sentence
	if current_chunk:
		chunks.append(current_chunk)
	translated_chunks = []
	for chunk in chunks:
		try:
			result = translator(chunk, max_length=max_chunk_size * 2)
			translated_chunks.append(result[0]['translation_text'])
		except Exception as e:
			logger.error(f"Translation error for chunk: {e}")
			translated_chunks.append(chunk)  # Fallback to original text
	return " ".join(translated_chunks)

def translate_batch(translator, texts: List[str], max_chunk_size: int = 512) -> List[str]:
    """Translate multiple texts efficiently by batching"""
    if not texts:
        return []

    # Combine texts with separators for batch processing
    separators = [" ||| " + str(i) + " ||| " for i in range(len(texts))]
    combined_text = texts[0]
    for i, text in enumerate(texts[1:], 1):
        combined_text += separators[i-1] + text

    try:
        # Translate the combined text
        result = translator(combined_text, max_length=max_chunk_size * len(texts))
        translated_combined = result[0]['translation_text']

        # Split back into individual translations
        translations = []
        parts = translated_combined.split(" ||| ")
        for part in parts:
            # Clean up separator artifacts
            clean_part = part.strip()
            for i in range(len(texts)):
                clean_part = clean_part.replace(f"{i} ||| ", "")
            translations.append(clean_part)

        # Ensure we return the right number of translations
        while len(translations) < len(texts):
            translations.append("")  # Fallback for missing translations

        return translations[:len(texts)]

    except Exception as e:
        logger.error(f"Batch translation failed: {e}")
        # Fallback to individual translation
        translations = []
        for text in texts:
            try:
                result = translator(text, max_length=max_chunk_size)
                translations.append(result[0]['translation_text'])
            except Exception as individual_error:
                logger.error(f"Individual translation failed: {individual_error}")
                translations.append(text)  # Fallback to original
        return translations

def add_user_credits(user_id: str, credits_to_add: int, description: str = "Credit purchase"):
	try:
		response = supabase.table("users").select("credits").eq("id", user_id).execute()
		if not response.data:
			raise Exception("User not found")
		current_credits = response.data[0]["credits"]
		new_credits = current_credits + credits_to_add
		supabase.table("users").update({"credits": new_credits}).eq("id", user_id).execute()
		transaction_id = str(uuid.uuid4())
		transaction_data = {
			"id": transaction_id,
			"user_id": user_id,
			"amount": credits_to_add,
			"type": "credit_purchase",
			"status": "completed",
			"description": description,
			"created_at": datetime.utcnow().isoformat()
		}
		supabase.table("transactions").insert(transaction_data).execute()
		logger.info(f"Added {credits_to_add} credits to user {user_id}. New balance: {new_credits}")
		return new_credits
	except Exception as credit_error:
		logger.error(f"Failed to add credits: {credit_error}")
		raise

def update_transaction_status(transaction_id: str, status: str, description: str = None):
	try:
		update_data = {
			"status": status,
			"updated_at": datetime.utcnow().isoformat()
		}
		if description:
			update_data["description"] = description
		supabase.table("transactions").update(update_data).eq("id", transaction_id).execute()
		logger.info(f"Updated transaction {transaction_id} to status: {status}")
		return True
	except Exception as update_error:
		logger.error(f"Failed to update transaction status: {update_error}")
		return False

def send_verification_email(email: str, name: str, verification_token: str):
	try:
		smtp_server = os.getenv("SMTP_HOST", "smtp.gmail.com")
		smtp_port = int(os.getenv("SMTP_PORT", "587"))
		smtp_username = os.getenv("SMTP_USER")
		smtp_password = os.getenv("SMTP_PASS")
		smtp_from = os.getenv("SMTP_FROM", "noreply@octavia.com")
		if not all([smtp_username, smtp_password]):
			logger.warning(f"SMTP credentials not configured. Mock email would be sent to: {email}")
			return True
		msg = MIMEMultipart('alternative')
		msg['Subject'] = "Verify Your Octavia Account"
		msg['From'] = smtp_from
		msg['To'] = email
		verification_link = f"http://localhost:3000/verify-email?token={verification_token}"
		html = f"""
		<!DOCTYPE html>
		<html>
		<head>
			<style>
				body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
				.container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
				.header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; color: white; border-radius: 10px 10px 0 0; }}
				.content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
				.button {{ display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 5px; font-weight: bold; }}
				.footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 12px; }}
			</style>
		</head>
		<body>
			<div class="container">
				<div class="header">
					<h1>Welcome to Octavia!</h1>
				</div>
				<div class="content">
					<h2>Hi {name},</h2>
					<p>Thank you for signing up for Octavia Video Translator. To start using your account, please verify your email address by clicking the button below:</p>
					<p style="text-align: center; margin: 30px 0;">
						<a href="{verification_link}" class="button">Verify Email Address</a>
					</p>
					<p>Or copy and paste this link into your browser:</p>
					<p style="background: #eee; padding: 10px; border-radius: 5px; word-break: break-all;">
						{verification_link}
					</p>
					<p>This link will expire in 24 hours.</p>
					<p>If you didn't create an account with Octavia, you can safely ignore this email.</p>
				</div>
				<div class="footer">
					<p>&copy; 2024 Octavia Video Translator. All rights reserved.</p>
				</div>
			</div>
		</body>
		</html>
		"""
		text = f"""Welcome to Octavia!
        
Hi {name},
        
Thank you for signing up for Octavia Video Translator. To start using your account, please verify your email address by clicking the link below:
        
{verification_link}
        
This link will expire in 24 hours.
        
If you didn't create an account with Octavia, you can safely ignore this email.
        
Best regards,
The Octavia Team"""
		part1 = MIMEText(text, 'plain')
		part2 = MIMEText(html, 'html')
		msg.attach(part1)
		msg.attach(part2)
		with smtplib.SMTP(smtp_server, smtp_port) as server:
			server.starttls()
			server.login(smtp_username, smtp_password)
			server.send_message(msg)
		logger.info(f"Verification email sent to {email}")
		return True
	except Exception as email_error:
		logger.error(f"Failed to send verification email to {email}: {email_error}")
		return True

def format_timestamp(seconds: float) -> str:
	"""Format seconds to SRT timestamp"""
	hours = int(seconds // 3600)
	minutes = int((seconds % 3600) // 60)
	secs = int(seconds % 60)
	millis = int((seconds - int(seconds)) * 1000)
	return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
