import psycopg2
import json
from dotenv import load_dotenv
import os
import binascii
import numpy as np
import streamlit as st
load_dotenv()
 
host = os.environ.get('host')
database = os.environ.get('database')
user = os.environ.get('user')
password = os.environ.get('password')
 
# print(host, database, user, password)
def get_connection():
    try:
        # Attempt to establish a connection
        conn = psycopg2.connect(
            host=host, database=database, user=user, password=password
        )
        print("Connection to PostgreSQL successful!")
        return conn
    except psycopg2.OperationalError as e:
        print("Error: Unable to connect to the database.")
        print(e)
        return None
 
def convert_numpy_images_to_binary(numpy_images):
    """Converts a list of NumPy arrays (images) to a list of binary strings."""
    binary_images = []
    for image_array in numpy_images:
        # Convert NumPy array to bytes
        binary_image = image_array.tobytes()
        binary_images.append(binary_image)
        return binary_images
 
 
def convert_images_to_binary(image_paths):
    binary_images = []
    for image_path in image_paths:
        with open(image_path, 'rb') as image_file:
            binary_images.append(image_file.read())
    return binary_images
 
def save_images_to_db(numpy_images):
    """Saves a list of NumPy arrays (images) to the database."""
    conn = None
    cursor = None
    try:
        for image in numpy_images:
            st.image(image, caption='image',channels="BGR")
        print("Establishing database connection...")
        conn = get_connection()
        if conn is None:
            raise Exception("Failed to establish a database connection.")
        cursor = conn.cursor()
        print("Database connection established.")
 
        # Convert NumPy images to binary strings
        print("Converting NumPy images to binary...")
        binary_images = convert_numpy_images_to_binary(numpy_images)
 
        # Convert the list of binary images to a JSON object
        print("Converting list of binary images to JSON...")
        binary_images_json = json.dumps([image.hex() for image in binary_images])
        print("Conversion successful.")
 
        # Create the table if it doesn't exist
        print("Creating table if it doesn't exist...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_binary (
                id SERIAL PRIMARY KEY,
                images JSONB NOT NULL
            )
        """)
        print("Table creation checked.")
 
        # Insert the binary images into the table
        print("Inserting binary images into the table...")
        cursor.execute("""
            INSERT INTO image_binary (images)
            VALUES (%s)
        """, [binary_images_json])
        conn.commit()
        print("Binary images inserted into database successfully.")
 
    except Exception as e:
        print(f"An error occurred: {e}")
 
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
        print("Database connection closed.")
 
def retrieve_images_from_db():
    """Retrieves images from the database and returns them as a list of NumPy arrays."""
    conn = None
    cursor = None
    try:
        print("Establishing database connection...")
        conn = get_connection()
        if conn is None:
            raise Exception("Failed to establish a database connection.")
        cursor = conn.cursor()
        print("Database connection established.")
 
    # Retrieve the binary images from the table
        print("Retrieving binary images from the table...")
        cursor.execute("SELECT images FROM image_binary")
        result = cursor.fetchone()
        if result is None or result[0] is None:
            raise Exception("No images found in the database.")
        binary_images_hex = result[0]
        print("Binary images retrieved successfully.")
 
        # Convert the list of hex strings back to binary images
        print("Converting list of hex strings back to binary images...")
        binary_images = [binascii.unhexlify(image_hex) for image_hex in binary_images_hex]
        for image in binary_images:
            print(binary_images)
        # Convert binary images back to NumPy arrays
        # numpy_images = []
        # for binary_image in binary_images:
        #     # Assuming you know the original shape and dtype of the images
        #     image_array = np.frombuffer(binary_image, dtype=np.uint8)  # Change dtype if needed
        #     image_array = image_array.reshape((100, 100, 3))  # Change reshape parameters if needed
        #     numpy_images.append(image_array)
 
        print("Conversion successful.")
        return binary_images
 
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
 
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
        print("Database connection closed.")
