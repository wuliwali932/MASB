"""
Medical Agent Orchestration System
Multi-agent LLM system for medical task automation using LangChain and LangGraph.
"""

from .orchestrator_agent import (
    OrchestratorAgent,
    TaskType,
    SubTask,
    TaskDecomposition
)

from .ehr_query_agent import (
    EHRQueryAgent,
    FHIRResourceType,
    FHIRQuery
)

from .external_tool_agent import (
    ExternalToolAgent,
    ToolCategory,
    ExternalTool
)

from .medical_calculation_agent import (
    MedicalCalculationAgent,
    RiskLevel,
    LabValueAssessment
)

from .multi_agent_orchestrator import (
    MultiAgentMedicalOrchestrator,
    AgentState,
    AgentType
)

__all__ = [
    # Orchestrator
    "OrchestratorAgent",
    "TaskType",
    "SubTask",
    "TaskDecomposition",
    # EHR Query
    "EHRQueryAgent",
    "FHIRResourceType",
    "FHIRQuery",
    # External Tools
    "ExternalToolAgent",
    "ToolCategory",
    "ExternalTool",
    # Medical Calculations
    "MedicalCalculationAgent",
    "RiskLevel",
    "LabValueAssessment",
    # Multi-Agent Orchestrator
    "MultiAgentMedicalOrchestrator",
    "AgentState",
    "AgentType"
]

__version__ = "1.0.0"
__author__ = "Medical AI Team"
__description__ = "Multi-agent LLM orchestration system for medical tasks"
