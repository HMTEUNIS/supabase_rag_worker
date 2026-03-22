-- One-shot setup: issue groups, threaded comments, and worker interpretation persistence.
-- Run in Supabase SQL editor as a single script.
-- After this: set Railway env GREATRX_INTERPRETATIONS_TABLE=interpretations (NOT knowledge_base).

begin;

-- 1) Issue groups (ids 101–105; skip if you already have these rows — ON CONFLICT does nothing)
create table if not exists public.issue_groups (
  id integer primary key,
  title text,
  status text default 'open',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

insert into public.issue_groups (id, name, status)
values
  (101, 'Issue group 101', 'open'),
  (102, 'Issue group 102', 'open'),
  (103, 'Issue group 103', 'open'),
  (104, 'Issue group 104', 'open'),
  (105, 'Issue group 105', 'open')
on conflict (id) do nothing;

-- 2) Comments linked to issue groups (worker reads from here for /api/rag/run-interpret)
create table if not exists public.issue_group_comments (
  id bigserial primary key,
  issue_group_id integer not null references public.issue_groups (id) on delete cascade,
  body text not null,
  author_id text,
  created_at timestamptz not null default now()
);

create index if not exists issue_group_comments_group_id_idx
  on public.issue_group_comments (issue_group_id);

create index if not exists issue_group_comments_created_at_idx
  on public.issue_group_comments (issue_group_id, created_at);

-- 3) Where the RAG worker stores each interpretation (must match rag/service.py insert payload)
create table if not exists public.interpretations (
  id uuid primary key default gen_random_uuid(),
  task text,
  external_ref jsonb not null,
  interpretation text not null,
  metadata jsonb,
  docs_used jsonb,
  confidence double precision,
  model text,
  processing_time_ms integer,
  created_at timestamptz not null default now()
);

create index if not exists interpretations_created_at_idx
  on public.interpretations (created_at desc);

create index if not exists interpretations_external_ref_gin_idx
  on public.interpretations using gin (external_ref jsonb_path_ops);

-- Optional: tie interpretation rows to issue_groups when external_ref contains issue_group_id
-- (no FK; external_ref is flexible for other tenants)

-- 4) Grants (worker uses service_role key)
grant usage on schema public to service_role;
grant select, insert, update, delete on public.issue_groups to service_role;
grant select, insert, update, delete on public.issue_group_comments to service_role;
grant select, insert, update, delete on public.interpretations to service_role;
grant usage, select on sequence public.issue_group_comments_id_seq to service_role;

-- 5) RLS (optional; service_role bypasses RLS in Supabase — safe for API worker)
alter table public.issue_groups enable row level security;
alter table public.issue_group_comments enable row level security;
alter table public.interpretations enable row level security;

-- Allow authenticated users to read/write their own app paths if you use anon key elsewhere;
-- adjust policies to your auth model. Example read-all for authenticated:
-- create policy "authenticated read issue_groups" on public.issue_groups for select to authenticated using (true);

commit;
