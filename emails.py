import streamlit as st
import smtplib
import io
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# st.title("Email Sender")

# Function to create email message
def create_email(from_email, to_email, subject, message, image_file, send_with_image):
    try:
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = to_email
        msg["Subject"] = subject

        if send_with_image and image_file:
            print("🔹 Processing image for attachment...")
            # Read image file
            img_bytes = io.BytesIO(image_file.read())
            image_data = img_bytes.getvalue()

            # Attach image as a separate file
            part = MIMEBase("application", "octet-stream")
            part.set_payload(image_data)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={image_file.name}")
            msg.attach(part)

            # Embed base64 image in email body
            encoded_string = base64.b64encode(image_data).decode()
            body = f"""
            <p>{message}</p>
            <p><b>Attached Image:</b></p>
            <img src="data:image/png;base64,{encoded_string}" width="300">
            """
            msg.attach(MIMEText(body, "html"))

            print("✅ Image attached successfully.")
        else:
            msg.attach(MIMEText(message, "plain"))

        return msg
    except Exception as e:
        print(f"Error creating email: {str(e)}")
        return None

# Function to send email
def send_email(from_email, password, to_email, msg):
    try:
        print("🔹 Connecting to SMTP server...")
        with smtplib.SMTP("smtp.gmail.com", 587) as server:  # Replace with correct SMTP
            server.starttls()
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
            st.success("✅ Email sent successfully!")
            # st.write("📨 Email sent!")
    except Exception as e:
        print(f"Error sending email: {str(e)}")

# # Streamlit UI Inputs
# from_email = st.text_input("Your Email", "your_email@example.com")
# password = st.text_input("Your Password", type="password")
# to_email = st.text_input("Recipient Email", "admin@example.com")
# subject = st.text_input("Subject", "Hello")
# message = st.text_area("Message", "This is a test email.")
# image_file = st.file_uploader("Upload an Image (Optional)", type=["png", "jpg", "jpeg"])

# # Option to send with or without an image
# send_with_image = st.radio("Send Mode", ["Message Only", "Message with Image"]) == "Message with Image"

# if st.button("Send Email"):
#     st.write("🔹 Preparing email...")
#     email_msg = create_email(from_email, to_email, subject, message, image_file, send_with_image)
#     if email_msg:
#         send_email(from_email, password, to_email, email_msg)
#     else:
#         st.error("❌ Email creation failed. Check logs.")