from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional


class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    role: Optional[str] = None


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = False
    role: str

class PatientInfo(BaseModel):
    id: str
    name: str
    age: int
    medical_history: List[str]

class MedicalRecord(BaseModel):
    # patient_id is supplied via the path param in the endpoint, make optional here
    patient_id: Optional[str] = None
    # Allow record_date to be optional (server may set it)
    record_date: Optional[datetime] = None
    diagnosis: str
    treatment: str
    notes: Optional[str] = None
    physician: Optional[str] = None
    # Support optional follow-up date used in tests
    follow_up_date: Optional[datetime] = None

class AppointmentRequest(BaseModel):
    # patient_id is implied by authenticated patient, make optional
    patient_id: Optional[str] = None
    preferred_date: datetime
    reason: str
    preferred_physician: Optional[str] = None
    urgency: str = "normal"  # normal, urgent, emergency


class UserIn(BaseModel):
    username: str
    password: str
