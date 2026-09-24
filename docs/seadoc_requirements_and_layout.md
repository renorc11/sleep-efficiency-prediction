# SeaDoc AI Requirements and Layout

## 1. Project Summary

SeaDoc AI is an offline-first medical assistant for maritime vessels. It helps ship crew and onboard medical staff access medical manual guidance, log patient vitals and incidents, draft structured reports, and sync/send reports when connectivity becomes available.

The product should prioritize:

- Offline usability on ships with unstable internet
- Fast local search over onboard medical manuals
- Secure local storage of sensitive medical data
- Low-bandwidth sync for deferred communication
- Clear workflows for emergencies and routine reporting

## 2. Recommended MVP Scope

The first version should focus on the smallest useful workflow:

- Local user login with role selection
- Vessel profile setup
- Offline PDF manual ingestion
- Local semantic/manual search over medical documents
- Guided first-aid query interface
- Patient encounter logging
- Vitals logging
- Structured medical report drafting
- Report queue for later sync/export
- Export to PDF or file bundle

Avoid in MVP:

- Microsoft Graph integration
- Multi-language UI
- Fleet analytics dashboards
- Live remote collaboration
- Training modules

## 3. Functional Requirements

### 3.1 User and Access

- Support local authentication
- Support roles:
  - Captain/Admin
  - Medical Officer
  - Crew Member
- Allow device-level session persistence
- Require explicit logout and local lock screen

### 3.2 Vessel Setup

- Store ship details:
  - Vessel name
  - IMO/identifier
  - Flag/state
  - Emergency contacts
  - Preferred reporting format
- Configure onboard medical manual library

### 3.3 Medical Manual Management

- Upload one or more PDF manuals
- Parse and chunk manuals locally
- Create embeddings locally where possible
- Store chunks in a local vector store
- Allow re-indexing after document update
- Show source citation for retrieved answers

### 3.4 Medical Guidance / RAG Assistant

- Accept natural-language medical queries
- Retrieve relevant manual sections
- Produce grounded answers from retrieved content
- Display:
  - answer summary
  - cited manual sections
  - emergency escalation warning when confidence is low
- Support quick prompts such as:
  - burns
  - fractures
  - poisoning
  - CPR
  - dehydration

### 3.5 Patient and Case Logging

- Create a patient/case record
- Capture:
  - patient name or crew ID
  - age
  - sex
  - rank/role
  - symptoms
  - incident time
  - location on vessel
  - notes
- Allow timeline updates for a case
- Allow attachment references to photos/files later

### 3.6 Vitals Logging

- Record vital signs:
  - temperature
  - pulse
  - blood pressure
  - respiration
  - oxygen saturation
  - pain score
- Show latest values and trend history
- Flag abnormal ranges visually

### 3.7 Report Generation

- Generate a structured medical incident report from case data
- Allow editable sections before finalizing
- Support report statuses:
  - draft
  - ready
  - queued
  - synced
  - exported
- Produce export formats:
  - PDF
  - JSON
  - compressed package for transfer

### 3.8 Offline Queue and Sync

- Queue outgoing reports while offline
- Retry sync when connectivity returns
- Compress payload before transmission
- Keep sync logs and retry status
- Allow manual export to file when email/API is unavailable

### 3.9 Audit and Safety

- Record who created/edited each case/report
- Timestamp all medical actions
- Prevent silent deletion of finalized reports
- Show disclaimer that guidance is manual-assisted, not a doctor replacement

## 4. Non-Functional Requirements

- Offline-first by default
- Works on standard CPU without GPU
- Usable on low-memory ship computers
- Local response for search/guidance should feel fast, ideally under 3 seconds for common queries
- Encrypt or protect local medical data at rest where feasible
- Clear recovery path after app restart/crash
- Minimal bandwidth usage during sync
- Installer-based distribution for Windows first, optional macOS/Linux later

## 5. Recommended Tech Stack

This stack is the best fit for the brief and later expansion:

- Desktop app shell: Tauri
- Frontend: React + TypeScript + Vite
- UI: Tailwind CSS + shadcn/ui or Mantine
- Local desktop backend: Rust in Tauri commands
- AI orchestration service: Python FastAPI sidecar
- Local database: SQLite
- Vector store: sqlite-vec for MVP
- LLM for local inference:
  - Phi-3 mini
  - or Llama 3.1 8B if target hardware can support it
- Embeddings:
  - bge-small-en
  - or nomic-embed-text local model
- RAG orchestration:
  - LlamaIndex for simpler document pipeline
  - or LangChain if you want broader integration options
- PDF parsing:
  - PyMuPDF
  - pypdf
- Report export:
  - WeasyPrint or pdfkit from HTML templates
- Sync/email later:
  - Microsoft Graph API
  - local fallback file export

## 6. System Modules

The application should be split into these modules:

- `auth`: local login, lock/unlock, roles
- `settings`: vessel profile, manuals, sync config
- `manuals`: PDF import, chunking, embedding, index management
- `assistant`: query processing, retrieval, grounded response generation
- `cases`: patient encounter records and timelines
- `vitals`: structured vital sign entries and trends
- `reports`: report drafting, approval, export, queue state
- `sync`: offline queue, retries, compression, remote dispatch
- `audit`: immutable activity trail

## 7. UI Layout Recommendation

### 7.1 App Structure

Use a desktop dashboard layout with:

- Left sidebar for primary navigation
- Top bar for vessel status, connectivity, and current user
- Main content area for work modules
- Right-side contextual panel only where helpful, such as citations or case summary

### 7.2 Navigation

Suggested sidebar items:

- Dashboard
- Medical Assistant
- Cases
- Vitals
- Reports
- Manuals
- Sync Queue
- Settings

### 7.3 Screen Layouts

#### Dashboard

Purpose:
- quick operational overview

Sections:
- connectivity status
- queued reports count
- open cases
- recent vital alerts
- shortcut cards for emergency topics

#### Medical Assistant

Layout:
- left: prompt/history panel
- center: chat/guidance response
- right: cited manual sections and source pages

Key actions:
- ask question
- view source
- add guidance to case note
- escalate to report

#### Cases

Layout:
- left: searchable case list
- center: selected case timeline
- right: patient summary and quick actions

Key actions:
- create case
- update symptoms
- attach vitals
- draft report

#### Vitals

Layout:
- top: patient/case selector
- center: vitals form
- bottom: trend cards/table

Key actions:
- add reading
- flag abnormal reading
- push reading into report

#### Reports

Layout:
- left: report list with status tabs
- center: editable structured report form
- right: export/sync panel

Key actions:
- generate from case
- edit
- approve/finalize
- export
- queue for sync

#### Manuals

Layout:
- top: upload/import actions
- center: manual library and indexing status
- right: selected manual metadata

Key actions:
- upload PDF
- re-index
- remove outdated version
- test retrieval

#### Sync Queue

Layout:
- table of pending/completed sync jobs

Key actions:
- retry
- cancel pending
- export manually
- inspect error details

## 8. Suggested Information Architecture

```text
SeaDoc AI
├── Dashboard
├── Medical Assistant
│   ├── Ask Query
│   ├── View Citations
│   └── Add to Case
├── Cases
│   ├── Case List
│   ├── Case Detail
│   ├── Symptoms and Notes
│   └── Linked Reports
├── Vitals
│   ├── New Entry
│   ├── Trend View
│   └── Alerts
├── Reports
│   ├── Drafts
│   ├── Ready
│   ├── Queued
│   └── Exported/Synced
├── Manuals
│   ├── Library
│   ├── Ingestion Status
│   └── Retrieval Test
├── Sync Queue
└── Settings
    ├── Vessel Profile
    ├── User Roles
    ├── Model Settings
    └── Export/Sync Config
```

## 9. Recommended Folder Layout

If we build this cleanly from scratch, use this structure:

```text
seadoc-ai/
├── apps/
│   ├── desktop/
│   │   ├── src/
│   │   │   ├── components/
│   │   │   ├── features/
│   │   │   │   ├── assistant/
│   │   │   │   ├── auth/
│   │   │   │   ├── cases/
│   │   │   │   ├── dashboard/
│   │   │   │   ├── manuals/
│   │   │   │   ├── reports/
│   │   │   │   ├── settings/
│   │   │   │   ├── sync/
│   │   │   │   └── vitals/
│   │   │   ├── hooks/
│   │   │   ├── layouts/
│   │   │   ├── lib/
│   │   │   ├── routes/
│   │   │   ├── store/
│   │   │   └── styles/
│   │   └── src-tauri/
│   │       ├── src/
│   │       │   ├── commands/
│   │       │   ├── db/
│   │       │   ├── models/
│   │       │   ├── services/
│   │       │   └── sync/
│   │       └── tauri.conf.json
│   └── ai-service/
│       ├── app/
│       │   ├── api/
│       │   ├── core/
│       │   ├── models/
│       │   ├── rag/
│       │   ├── schemas/
│       │   ├── services/
│       │   └── workers/
│       ├── tests/
│       └── requirements.txt
├── packages/
│   ├── shared-types/
│   └── ui/
├── data/
│   ├── manuals/
│   ├── exports/
│   ├── queue/
│   └── sqlite/
├── docs/
│   ├── requirements/
│   ├── ux/
│   └── architecture/
├── scripts/
└── docker/
```

## 10. Core Database Tables

Minimum SQLite tables:

- `users`
- `vessels`
- `manual_documents`
- `manual_chunks`
- `embedding_index_meta`
- `cases`
- `case_notes`
- `vitals`
- `reports`
- `report_exports`
- `sync_jobs`
- `audit_logs`

## 11. Suggested MVP User Flow

1. Admin configures vessel and uploads manuals.
2. Crew member or medical officer opens SeaDoc AI offline.
3. User asks medical guidance question.
4. System retrieves manual evidence and shows grounded answer.
5. User creates or opens a case.
6. User logs symptoms and vitals.
7. User generates a structured report draft.
8. User exports locally or queues the report for later sync.

## 12. Delivery Phases

### Phase 1

- Desktop shell
- Local auth
- Vessel settings
- Case logging
- Vitals logging
- Report drafting
- Local export

### Phase 2

- PDF ingestion
- Embeddings and local vector search
- RAG assistant with citations

### Phase 3

- Offline queue
- Compressed sync
- Remote dispatch integration

### Phase 4

- Multilingual UI
- Analytics
- Training mode
- Multi-vessel management

## 13. Build Recommendation

The most practical build order is:

1. Create the Tauri + React desktop shell and SQLite schema.
2. Build case, vitals, and report workflows first.
3. Add manual ingestion and local RAG once the core data model is stable.
4. Add sync/export integrations after offline workflows are solid.

This keeps the first usable version simple, demonstrable, and aligned with the original SeaDoc AI concept.
