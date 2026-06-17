-- Run once in Supabase SQL Editor (fixes email-only signups)

ALTER TABLE chat_users ALTER COLUMN phone_number DROP NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS chat_users_email_unique
    ON chat_users (email)
    WHERE email IS NOT NULL;
