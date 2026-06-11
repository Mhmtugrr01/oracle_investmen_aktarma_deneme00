-- Oracle Master-Swarm V4.0 — Supabase Schema
-- Bu SQL'i Supabase SQL Editor'da çalıştırın

-- Oracle görevleri tablosu
CREATE TABLE IF NOT EXISTS oracle_tasks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id TEXT NOT NULL,
    user_input TEXT,
    expanded_prompt TEXT,
    agent TEXT,
    result TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- İndeksler
CREATE INDEX IF NOT EXISTS idx_oracle_tasks_user_id ON oracle_tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_oracle_tasks_created_at ON oracle_tasks(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_oracle_tasks_agent ON oracle_tasks(agent);

-- Quant analiz geçmişi
CREATE TABLE IF NOT EXISTS quant_analyses (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id TEXT,
    symbol TEXT,
    price DECIMAL,
    rsi DECIMAL,
    signal TEXT,
    confidence INTEGER,
    raw_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quant_analyses_symbol ON quant_analyses(symbol);
CREATE INDEX IF NOT EXISTS idx_quant_analyses_created_at ON quant_analyses(created_at DESC);

-- Marketing leads tablosu
CREATE TABLE IF NOT EXISTS marketing_leads (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id TEXT,
    company TEXT,
    contact_email TEXT,
    sector TEXT,
    location TEXT,
    email_body TEXT,
    status TEXT DEFAULT 'draft',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Row Level Security (RLS)
ALTER TABLE oracle_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE quant_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE marketing_leads ENABLE ROW LEVEL SECURITY;

-- Service role her şeye erişebilir
CREATE POLICY IF NOT EXISTS "service_role_all" ON oracle_tasks
    FOR ALL USING (true);
CREATE POLICY IF NOT EXISTS "service_role_all" ON quant_analyses
    FOR ALL USING (true);
CREATE POLICY IF NOT EXISTS "service_role_all" ON marketing_leads
    FOR ALL USING (true);

-- pgvector extension (uzun dönem hafıza için)
CREATE EXTENSION IF NOT EXISTS vector;

-- Vektör bellek tablosu (gelecek versiyon)
CREATE TABLE IF NOT EXISTS oracle_memory (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id TEXT,
    content TEXT,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_oracle_memory_embedding
    ON oracle_memory USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
