-- ===========================================================================
-- ShasanAI Sovereign Authentication & Chat History Schema Migration (002_auth_schema.sql)
-- Multi-Tenant Officer Persona Security and Session Isolation
-- ===========================================================================

-- 1. Users & Officer Accounts Table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    full_name TEXT NOT NULL,
    department TEXT NOT NULL,
    designation TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'OFFICER', -- 'OFFICER', 'ADMIN', 'AUDITOR'
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_dept ON users(department);

-- 2. Chat Sessions Table (Persistent Thread / History Linking)
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    department TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated ON chat_sessions(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_dept ON chat_sessions(department);

-- 3. Seed Uttarakhand Secretariat Pre-configured Officer Personas
-- Default evaluation password: 'Shasan@2026' (bcrypt hashed)
INSERT INTO users (email, hashed_password, full_name, department, designation, role)
VALUES 
    (
        'forest.officer@uk.gov.in',
        '$2b$12$5OJ5L17RIdcvCbDX5E8DiebsAPtOV4wsbtXTZT0t9vbx9rre2vpni',
        'Vikram Singh Negi',
        'Forest',
        'Divisional Forest Officer (DFO)',
        'OFFICER'
    ),
    (
        'finance.officer@uk.gov.in',
        '$2b$12$5OJ5L17RIdcvCbDX5E8DiebsAPtOV4wsbtXTZT0t9vbx9rre2vpni',
        'Pooja Sharma',
        'Finance',
        'Senior Accounts Officer',
        'OFFICER'
    ),
    (
        'personnel.officer@uk.gov.in',
        '$2b$12$5OJ5L17RIdcvCbDX5E8DiebsAPtOV4wsbtXTZT0t9vbx9rre2vpni',
        'Rajesh Chandra',
        'Personnel',
        'Joint Secretary (Personnel)',
        'OFFICER'
    ),
    (
        'admin.itda@uk.gov.in',
        '$2b$12$5OJ5L17RIdcvCbDX5E8DiebsAPtOV4wsbtXTZT0t9vbx9rre2vpni',
        'Amitabh Rawat',
        'General',
        'Director (ITDA)',
        'ADMIN'
    )
ON CONFLICT (email) DO UPDATE SET
    hashed_password = EXCLUDED.hashed_password,
    full_name = EXCLUDED.full_name,
    department = EXCLUDED.department,
    designation = EXCLUDED.designation,
    role = EXCLUDED.role;
