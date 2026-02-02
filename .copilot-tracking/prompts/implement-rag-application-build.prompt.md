---
mode: agent
model: Claude Sonnet 4
---

<!-- markdownlint-disable-file -->

# Implementation Prompt: RAG Application Build

## Implementation Instructions

### Step 1: Create Changes Tracking File

You WILL create `20260202-rag-application-build-changes.md` in #file:../changes/ if it does not exist.

### Step 2: Execute Implementation

You WILL follow #file:../../.github/instructions/task-implementation.instructions.md if it exists, otherwise follow these instructions:

You WILL systematically implement #file:../plans/20260202-rag-application-build-plan.instructions.md task-by-task

**Implementation Guidelines:**

1. **Follow the Plan**: Implement each phase and task in the order specified
2. **Check Off Tasks**: Mark tasks as `[x]` in the plan file as you complete them
3. **Document Changes**: Record all changes in the changes tracking file
4. **Follow Standards**: Adhere to #file:../../.github/python.instructions for all Python code
5. **Test As You Go**: Validate each component before moving to the next
6. **Reference Research**: Consult #file:../research/20260202-rag-application-build-research.md for implementation details
7. **Reference Details**: Consult #file:../details/20260202-rag-application-build-details.md for specific requirements

**CRITICAL**: If ${input:phaseStop:true} is true, you WILL stop after each Phase for user review.
**CRITICAL**: If ${input:taskStop:false} is true, you WILL stop after each Task for user review.

### Step 3: Testing and Validation

After implementing each phase:

1. **Unit Tests**: Write and run tests for the components
2. **Integration Tests**: Verify components work together
3. **Manual Testing**: Test functionality through the UI or API
4. **Performance Check**: Ensure performance requirements are met

### Step 4: Documentation

As you implement:

1. **Code Comments**: Add clear comments following Python guidelines
2. **Docstrings**: Include comprehensive docstrings for all functions and classes
3. **README Updates**: Keep documentation synchronized with implementation
4. **Change Log**: Maintain detailed change log in tracking file

### Step 5: Cleanup

When ALL Phases are checked off (`[x]`) and completed you WILL do the following:

1. You WILL provide a markdown style link and a summary of all changes from #file:../changes/20260202-rag-application-build-changes.md to the user:

   - You WILL keep the overall summary brief
   - You WILL add spacing around any lists
   - You MUST wrap any reference to a file in a markdown style link

2. You WILL provide markdown style links to .copilot-tracking/plans/20260202-rag-application-build-plan.instructions.md, .copilot-tracking/details/20260202-rag-application-build-details.md, and .copilot-tracking/research/20260202-rag-application-build-research.md documents. You WILL recommend cleaning these files up as well.

3. **MANDATORY**: You WILL attempt to delete .copilot-tracking/prompts/implement-rag-application-build.prompt.md

## Implementation Phases Summary

### Phase 1: Foundation Setup
Create project structure, install dependencies, configure settings, initialize database

### Phase 2: Document Processing Pipeline
Implement file upload, PDF/Excel extraction, chunking, and metadata management

### Phase 3: Vector Store Integration
Set up ChromaDB, embedding generation, vector storage and retrieval operations

### Phase 4: RAG Pipeline Implementation
Build query processing, context building, prompt construction, LLM integration, citations

### Phase 5: REST API Development
Create Flask app with blueprints, implement /api/upload, /api/chat, /api/documents endpoints

### Phase 6: Web User Interface
Develop HTML templates, chat interface, upload interface, document management UI, sources panel

### Phase 7: Testing and Validation
Write unit tests, integration tests, end-to-end tests, perform validation

### Phase 8: Documentation and Deployment
Create README, deployment guides, configuration templates, Docker support

## Key Considerations

### Security
- Validate all user inputs
- Sanitize file uploads
- Use environment variables for secrets
- Implement proper error handling
- Follow security best practices

### Performance
- Optimize vector search queries
- Implement caching where appropriate
- Use batch processing for embeddings
- Monitor response times
- Profile and optimize bottlenecks

### Maintainability
- Follow PEP 8 style guide
- Write clear, documented code
- Create modular, reusable components
- Maintain comprehensive tests
- Keep documentation updated

### Reliability
- Implement error handling throughout
- Add logging for debugging
- Create health check endpoints
- Handle edge cases gracefully
- Test failure scenarios

## Success Criteria

- [ ] Changes tracking file created and maintained
- [ ] All plan items implemented with working code
- [ ] All detailed specifications satisfied
- [ ] Project conventions followed (#file:../../.github/python.instructions)
- [ ] Tests written and passing (>80% coverage)
- [ ] Documentation complete and accurate
- [ ] Application deployable and functional
- [ ] Performance requirements met
- [ ] Security best practices applied

## Resources

- **Research**: #file:../research/20260202-rag-application-build-research.md
- **Plan**: #file:../plans/20260202-rag-application-build-plan.instructions.md
- **Details**: #file:../details/20260202-rag-application-build-details.md
- **SPARC Docs**: SPARC_Documents/ (Specification, Architecture, Pseudocode, Refinement, Completion)
- **Python Standards**: #file:../../.github/python.instructions
- **Changes**: #file:../changes/20260202-rag-application-build-changes.md (to be created)
