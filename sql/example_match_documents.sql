-- Example RPC for pgvector similarity search via Supabase PostgREST.
-- 1) Replace `1536` with your embedding dimension (must match EMBEDDING_DIMENSION + column type).
-- 2) Replace `your_docs` and column names with your table.
-- 3) Grant execute to `service_role` (or anon/authenticated as appropriate).
-- 4) Set env GREATRX_VECTOR_RPC=match_documents (or whatever you name the function).

create extension if not exists vector;

-- Example table (optional — use your own schema)
-- create table public.your_docs (
--   id bigint generated always as identity primary key,
--   content text not null,
--   embedding vector(1536) not null,
--   category text,
--   tags text[]
-- );
-- create index on public.your_docs using ivfflat (embedding vector_cosine_ops);

create or replace function public.match_documents(
  p_query_embedding vector(1536),
  p_match_count int default 5,
  p_category text default null,
  p_tags text[] default null
)
returns table (
  id bigint,
  content text,
  similarity double precision
)
language sql
stable
as $$
  select
    d.id,
    d.content,
    (1 - (d.embedding <=> p_query_embedding))::double precision as similarity
  from public.your_docs d
  where
    (p_category is null or d.category = p_category)
    and (p_tags is null or d.tags && p_tags)
  order by d.embedding <=> p_query_embedding
  limit greatest(1, least(p_match_count, 50));
$$;

grant execute on function public.match_documents(vector, int, text, text[]) to service_role;
