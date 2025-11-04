# QA Testing Checklist - Ollama Web GUI Frontend

## Overview
This checklist ensures comprehensive testing of all frontend features before production deployment.

---

## 1. Setup & Authentication Flow

### Initial Setup
- [ ] First-time user sees setup screen
- [ ] Ollama URL input accepts valid URLs
- [ ] Ollama URL validation rejects invalid formats
- [ ] API Key input field shows/hides password
- [ ] Test Connection button disabled without valid inputs
- [ ] Test Connection shows loading state during test
- [ ] Successful connection shows success message with model count
- [ ] Failed connection shows clear error message
- [ ] Retry button appears on connection failure
- [ ] Configuration saves to localStorage
- [ ] User redirects to chat page after successful setup

### Re-authentication
- [ ] Logout button clears authentication state
- [ ] Logout redirects to setup page
- [ ] User can reconfigure from settings button
- [ ] Protected routes redirect to setup when not authenticated

---

## 2. Chat Interface

### Layout & Responsiveness
- [ ] Three-part layout displays correctly (header, sidebar, chat area)
- [ ] Layout responsive on mobile (320px - 767px)
- [ ] Layout responsive on tablet (768px - 1023px)
- [ ] Layout responsive on desktop (1024px+)
- [ ] Sidebar collapses on mobile with hamburger menu
- [ ] Sidebar drawer works on mobile
- [ ] Header stays fixed on scroll

### Conversation Management
- [ ] "New Chat" button creates new conversation
- [ ] Conversation list shows all conversations
- [ ] Active conversation is highlighted
- [ ] Clicking conversation loads messages
- [ ] Delete button shows confirmation modal
- [ ] Deleting conversation removes it from list
- [ ] Search/filter conversations works
- [ ] Conversation shows last message preview
- [ ] Conversation shows model name
- [ ] Empty state shows when no conversations

---

## 3. E2E Tests
- [ ] Setup flow E2E test passes
- [ ] Chat flow E2E test passes
- [ ] Export/Import E2E test passes

---

**Status:** Ready for Testing
**Last Updated:** November 2025
