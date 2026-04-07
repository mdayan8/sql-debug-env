"""
Typed Pydantic models for the SQL Debug Environment.
Implements the OpenEnv spec: Observation, Action, Reward.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ActionType(str, Enum):
    SUBMIT_QUERY = "submit_query"       # Submit a fixed SQL query for evaluation
    INSPECT_SCHEMA = "inspect_schema"  # Request schema info (costs 0 reward, gives info)
    INSPECT_ERROR = "inspect_error"    # Request error details (costs 0, gives stack trace)
    INSPECT_SAMPLE = "inspect_sample"  # Request 3 sample rows from a table
    RESET_QUERY = "reset_query"        # Reset to the original broken query (costs -0.05 penalty)


class SQLDebugAction(BaseModel):
    """
    Action model for the SQL Debug Environment.

    The agent can either:
    - submit_query: Submit a fixed SQL string for evaluation
    - inspect_schema: Get table schema info (free action, no reward change)
    - inspect_error: Get detailed error message from last query run
    - inspect_sample: Get sample rows from a specified table
    - reset_query: Go back to original broken query (costs -0.05 penalty)
    """
    action_type: ActionType = Field(
        description="Type of action to take"
    )
    query: Optional[str] = Field(
        default=None,
        description="SQL query string. Required when action_type is 'submit_query'."
    )
    table_name: Optional[str] = Field(
        default=None,
        description="Table name. Required when action_type is 'inspect_sample'."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "action_type": "submit_query",
                "query": "SELECT u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name ORDER BY order_count DESC"
            }
        }


class QueryResult(BaseModel):
    """Result of executing a SQL query."""
    success: bool
    rows: Optional[List[Dict[str, Any]]] = None
    row_count: Optional[int] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None


class SchemaInfo(BaseModel):
    """Database schema information."""
    tables: Dict[str, List[Dict[str, str]]]  # table_name -> list of {name, type, nullable}
    sample_data: Optional[Dict[str, List[Dict[str, Any]]]] = None


class SQLDebugObservation(BaseModel):
    """
    Observation returned after each step.

    Contains the current state of the debugging session:
    - The original broken query (always visible)
    - The agent's current best query
    - Result of last action
    - Progress indicators
    - Schema/error info if requested
    """
    task_id: str = Field(description="Current task identifier")
    task_description: str = Field(description="Natural language description of the bug to fix")
    original_query: str = Field(description="The original broken SQL query")
    current_query: Optional[str] = Field(default=None, description="Agent's last submitted query")
    expected_description: str = Field(description="Description of what the correct output should look like")

    # Last action result
    last_action_type: str
    last_query_result: Optional[QueryResult] = None

    # Progress
    steps_taken: int
    steps_remaining: int
    current_score: float = Field(description="Current score 0.0-1.0 for this episode")

    # Contextual help (populated based on action type)
    schema_info: Optional[SchemaInfo] = None
    error_details: Optional[str] = None
    sample_rows: Optional[List[Dict[str, Any]]] = None

    # Hints (unlocked after step 3 on easy, step 5 on medium/hard)
    hint: Optional[str] = None

    # Episode status
    is_done: bool = False
    success: bool = False


class SQLDebugReward(BaseModel):
    """
    Reward signal for the SQL Debug Environment.

    Reward components (all sum to final reward):
    - correctness: 0.0-0.6 based on row-level match vs expected output
    - efficiency: 0.0-0.2 bonus for solving in fewer steps
    - syntax_progress: 0.0-0.1 for getting a syntactically valid query (even if wrong)
    - schema_bonus: 0.0-0.1 for queries that reference correct tables/columns
    - penalties: negative values for reset_query, infinite loops, destructive SQL
    """
    value: float = Field(ge=0.0, le=1.0, description="Total reward for this step")
    correctness: float = Field(ge=0.0, le=0.6)
    efficiency: float = Field(ge=0.0, le=0.2)
    syntax_progress: float = Field(ge=0.0, le=0.1)
    schema_bonus: float = Field(ge=0.0, le=0.1)
    penalty: float = Field(ge=0.0, le=0.2, description="Penalty deduction magnitude (non-negative)")
    breakdown: str = Field(description="Human-readable reward breakdown")


class EpisodeState(BaseModel):
    """Full internal state of an episode. Used by state() endpoint."""
    task_id: str
    task_difficulty: str
    original_query: str
    current_query: Optional[str]
    best_score_so_far: float
    steps_taken: int
    max_steps: int
    action_history: List[Dict[str, Any]]
    reward_history: List[float]
    is_done: bool
    success: bool
    db_schema: Dict[str, Any]

