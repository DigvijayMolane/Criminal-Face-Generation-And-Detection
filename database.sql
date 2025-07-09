-- Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS criminal_detection;
USE criminal_detection;

-- Users table
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Criminals table
CREATE TABLE criminals (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    age INT,
    gender VARCHAR(50),
    crime TEXT,
    image_path VARCHAR(255),
    facial_embedding MEDIUMBLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_criminals_name ON criminals(name);
CREATE INDEX idx_users_username ON users(username);

INSERT INTO users (username, password) 
VALUES ('admin', 'pbkdf2:sha256:260000$YOUR_HASHED_PASSWORD');