"""
DIFF — research_platform/mediated_services.py
─────────────────────────────────────────────
This file documents the ONLY change required in mediated_services.py.
Apply it by replacing the _compose_description method in
SimulationDesignerService.

FIND (the entire method, ~25 lines):

    def _compose_description(self, task: TaskEnvelope, sources: list[ArtifactRef]) -> str:
        parts = [task.instructions.strip()]
        selected_targets = task.metadata.get("selected_simulation_targets")
        if isinstance(selected_targets, list) and selected_targets:
            ...
        context_bits = []
        for source in sources:
            if source.kind == "article_brief":
                text = self._registry.read_artifact_text(source.artifact_id) or json.dumps(source.metadata, indent=2)
                context_bits.append(f"Article brief:\n{text}")
            elif source.kind == "literature_review":
                context_bits.append(f"Literature synthesis summary:\n{source.summary}")
            elif source.summary:
                context_bits.append(f"{source.title}:\n{source.summary}")
        if context_bits:
            parts.append("Context artifacts:\n" + "\n\n".join(context_bits))
        return "\n\n".join(part for part in parts if part.strip())

REPLACE WITH:

    def _compose_description(self, task: TaskEnvelope, sources: list[ArtifactRef]) -> str:
        from sim_tool.description_builder import build_designer_description
        return build_designer_description(
            task_instructions=task.instructions,
            sources=sources,
            task_metadata=dict(task.metadata or {}),
        )

That's the complete change. The description_builder module handles all
the precision-ordering and length capping.
"""
