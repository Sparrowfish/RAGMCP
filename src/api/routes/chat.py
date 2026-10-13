# src/api/routes/chat.py
from fastapi import APIRouter, HTTPException
from loguru import logger
from src.api.schemas import ChatRequest, ChatResponse, SourceDocument
from src.api.dependencies import state
from src.model.inference import generate
from src.rag.pipeline import retrieve_context
from src.mcp.integration import LLMChatRouter

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    cfg = state.cfg
    top_k = request.top_k or cfg["rag"]["top_k"]
    sources: list[SourceDocument] = []
    context = ""
    rag_used = False

    # Continue confirmation / argument collection without re-routing the turn.
    if request.tool_session_id:
        if not state.mcp_loaded:
            raise HTTPException(status_code=503, detail="MCP tools are not configured.")
        if request.tool_confirmation is None:
            tool_response = await state.mcp_engine.handle(request.tool_session_id, request.message)
        else:
            tool_response = await state.mcp_engine.confirm(request.tool_session_id, request.tool_confirmation)
        return ChatResponse(
            response=tool_response.message,
            sources=[],
            rag_used=False,
            tokens_generated=0,
            tool_used=True,
            tool_name=tool_response.tool_name,
            tool_session_id=tool_response.session_id,
            tool_state=tool_response.state,
            tool_result=tool_response.result,
        )

    # Let the model choose the information source before retrieving or acting.
    route_kind = "chat"
    if request.use_tools and state.mcp_loaded:
        route = LLMChatRouter(state.model, state.tokenizer, state.mcp_engine.registry).route(
            request.message, rag_available=request.use_rag and state.vectorstore_loaded,
        )
        route_kind = route.kind
        if route.kind == "tool":
            tool_response = await state.mcp_engine.start(
                request.user_id, route.tool_name, request.message,
            )
            return ChatResponse(
                response=tool_response.message,
                sources=[],
                rag_used=False,
                tokens_generated=0,
                tool_used=True,
                tool_name=tool_response.tool_name,
                tool_session_id=tool_response.session_id,
                tool_state=tool_response.state,
                tool_result=tool_response.result,
            )
    elif request.use_rag and state.vectorstore_loaded:
        # Preserve retrieval for deployments where MCP has not been enabled yet.
        route_kind = "rag"

    # --- RAG retrieval ---
    if route_kind == "rag" and request.use_rag and state.vectorstore_loaded:
        try:
            raw_results = state.vectorstore.search(request.message, top_k=top_k)
            context = retrieve_context(request.message, state.vectorstore, top_k=top_k)
            sources = [
                SourceDocument(
                    content=r["content"][:300],  # truncate for response payload
                    source=r["source"],
                    score=r["score"],
                )
                for r in raw_results
            ]
            rag_used = True
            logger.info(f"RAG retrieved {len(sources)} chunks for query: '{request.message[:60]}...'")
        except Exception as e:
            logger.warning(f"RAG retrieval failed, falling back to no context: {e}")

    elif route_kind == "rag" and request.use_rag and not state.vectorstore_loaded:
        logger.warning("RAG requested but vector store not loaded. Run ingestion first.")

    # --- Generation ---
    try:
        response_text = generate(
            model=state.model,
            tokenizer=state.tokenizer,
            user_message=request.message,
            system_prompt=request.system_prompt or "You are a helpful assistant.",
            context=context,
            max_new_tokens=request.max_new_tokens or 512,
            temperature=request.temperature or 0.7,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

    # Rough token count from the tokenizer
    tokens_generated = len(state.tokenizer.encode(response_text))

    return ChatResponse(
        response=response_text,
        sources=sources,
        rag_used=rag_used,
        tokens_generated=tokens_generated,
    )
