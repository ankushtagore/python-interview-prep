## **📋 TABLE OF CONTENTS**

1. [FastAPI Fundamentals (1-30)](#fastapi-fundamentals)
2. [Advanced FastAPI Features (31-60)](#advanced-fastapi-features)
3. [Database & ORM (61-90)](#database--orm)
4. [Authentication & Security (91-120)](#authentication--security)
5. [Async Programming (121-150)](#async-programming)
6. [Testing & Quality Assurance (151-180)](#testing--quality-assurance)
7. [Performance & Optimization (181-200)](#performance--optimization)
8. [Architecture & Design Patterns (201-230)](#architecture--design-patterns)

---

## **1. FASTAPI FUNDAMENTALS**

### **Q1: What is FastAPI and what are its key advantages?**

**Answer:**
FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.

**Key Advantages:**
- **High Performance**: One of the fastest Python frameworks available, comparable to NodeJS and Go
- **Type Safety**: Built-in support for Python type hints with automatic validation
- **Auto Documentation**: Automatic interactive API documentation (Swagger UI/ReDoc)
- **Modern Python**: Uses modern Python features like async/await
- **Standards Based**: Based on OpenAPI and JSON Schema standards
- **Easy to Use**: Designed to be easy to use and learn

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = False

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/items/")
async def create_item(item: Item):
    return item
```

### **Q2: How do you create a basic FastAPI application with proper structure?**

**Answer:**
Based on the NeuroShiksha codebase structure:

```python
# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from .database import init_db, cleanup_database
from .routes import auth_router, courses_router, users_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown
    await cleanup_database()

app = FastAPI(
    title="NeuroShiksha API",
    version="3.0.0",
    lifespan=lifespan,
    description="Educational platform API"
)

# Include routers
app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
app.include_router(courses_router, prefix="/api/courses", tags=["courses"])
app.include_router(users_router, prefix="/api/users", tags=["users"])
```

### **Q3: Explain the difference between FastAPI and Flask/Django.**

**Answer:**

| Feature | FastAPI | Flask | Django |
|---------|---------|-------|--------|
| **Performance** | Very High (async) | Medium | Medium |
| **Type Safety** | Built-in | Manual | Manual |
| **Documentation** | Auto-generated | Manual | Manual |
| **Async Support** | Native | Limited | Limited |
| **Learning Curve** | Easy | Easy | Steep |
| **Use Case** | APIs, Microservices | Web apps, APIs | Full web apps |
| **Validation** | Automatic | Manual | Manual |

### **Q4: How do you handle different HTTP methods in FastAPI?**

**Answer:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

# GET - Retrieve data
@app.get("/items/{item_id}")
async def get_item(item_id: int):
    return {"item_id": item_id, "name": "Sample Item"}

# POST - Create new resource
@app.post("/items/")
async def create_item(item: Item):
    return {"message": "Item created", "item": item}

# PUT - Update entire resource
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"message": f"Item {item_id} updated", "item": item}

# PATCH - Partial update
@app.patch("/items/{item_id}")
async def partial_update_item(item_id: int, item: dict):
    return {"message": f"Item {item_id} partially updated"}

# DELETE - Remove resource
@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    return {"message": f"Item {item_id} deleted"}
```

### **Q5: What are Pydantic models and how do they work in FastAPI?**

**Answer:**
Pydantic models provide data validation and serialization using Python type annotations.

```python
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"

class UserCreate(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.STUDENT
    
    @validator('password')
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        return v

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: UserRole
    created_at: datetime
    
    class Config:
        from_attributes = True  # For SQLAlchemy models
```

### **Q6: How do you handle path parameters and query parameters?**

**Answer:**
```python
from fastapi import FastAPI, Query, Path
from typing import Optional, List

app = FastAPI()

# Path parameters
@app.get("/users/{user_id}")
async def get_user(user_id: int = Path(..., gt=0, description="User ID")):
    return {"user_id": user_id}

# Query parameters
@app.get("/items/")
async def get_items(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of items to return"),
    search: Optional[str] = Query(None, min_length=1, max_length=50),
    tags: List[str] = Query([], description="Filter by tags")
):
    return {
        "skip": skip,
        "limit": limit,
        "search": search,
        "tags": tags
    }

# Multiple path parameters
@app.get("/users/{user_id}/courses/{course_id}")
async def get_user_course(
    user_id: int = Path(..., gt=0),
    course_id: int = Path(..., gt=0)
):
    return {"user_id": user_id, "course_id": course_id}
```

### **Q7: How do you handle request and response models?**

**Answer:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class CourseCreate(BaseModel):
    title: str
    description: str
    difficulty_level: int = 1

class CourseUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    difficulty_level: Optional[int] = None

class CourseResponse(BaseModel):
    id: str
    title: str
    description: str
    difficulty_level: int
    created_at: datetime
    is_published: bool = False

class CourseListResponse(BaseModel):
    courses: List[CourseResponse]
    total: int
    page: int
    per_page: int

@app.post("/courses/", response_model=CourseResponse)
async def create_course(course: CourseCreate):
    # Create course logic
    return CourseResponse(
        id="123",
        title=course.title,
        description=course.description,
        difficulty_level=course.difficulty_level,
        created_at=datetime.now()
    )

@app.get("/courses/", response_model=CourseListResponse)
async def list_courses():
    return CourseListResponse(
        courses=[],
        total=0,
        page=1,
        per_page=10
    )
```

### **Q8: How do you handle HTTP status codes and exceptions?**

**Answer:**
```python
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    if user_id < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID must be positive"
        )
    
    user = await find_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user

# Custom exception handler
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)}
    )

# Custom response
@app.post("/courses/", status_code=status.HTTP_201_CREATED)
async def create_course(course: CourseCreate):
    return {"message": "Course created successfully"}
```

### **Q9: How do you implement middleware in FastAPI?**

**Answer:**
```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "*.example.com"]
)

# Custom middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Authentication middleware
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/protected"):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid authorization header"}
            )
    response = await call_next(request)
    return response
```

### **Q10: How do you organize FastAPI applications with routers?**

**Answer:**
Based on the NeuroShiksha structure:

```python
# app/routes/__init__.py
from fastapi import APIRouter
from .auth import router as auth_router
from .courses import router as courses_router
from .users import router as users_router

# app/routes/auth.py
from fastapi import APIRouter, Depends, HTTPException
from ..schemas.user import UserCreate, UserResponse
from ..services.user_service import UserService

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate, user_service: UserService = Depends(get_user_service)):
    return await user_service.create_user(user_data)

@router.post("/login")
async def login(credentials: LoginRequest):
    # Login logic
    pass

# app/routes/courses.py
from fastapi import APIRouter, Depends
from ..schemas.course import CourseCreate, CourseResponse
from ..services.course_service import CourseService

router = APIRouter(prefix="/courses", tags=["courses"])

@router.get("/", response_model=List[CourseResponse])
async def list_courses(course_service: CourseService = Depends(get_course_service)):
    return await course_service.get_all_courses()

# app/main.py
from fastapi import FastAPI
from .routes import auth_router, courses_router, users_router

app = FastAPI()
app.include_router(auth_router)
app.include_router(courses_router)
app.include_router(users_router)
```

---

## **2. ADVANCED FASTAPI FEATURES**

### **Q11: How do you implement dependency injection in FastAPI?**

**Answer:**
```python
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Generator

app = FastAPI()

# Database dependency
async def get_db() -> Generator[AsyncSession, None, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Service dependencies
def get_user_service(db: AsyncSession = Depends(get_db)) -> UserService:
    return UserService(db)

def get_course_service(db: AsyncSession = Depends(get_db)) -> CourseService:
    return CourseService(db)

# Authentication dependency
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials"
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = await get_user(user_id)
    if user is None:
        raise credentials_exception
    return user

# Using dependencies
@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/courses/")
async def create_course(
    course: CourseCreate,
    current_user: User = Depends(get_current_user),
    course_service: CourseService = Depends(get_course_service)
):
    return await course_service.create_course(course, current_user.id)
```

### **Q12: How do you implement background tasks in FastAPI?**

**Answer:**
```python
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import asyncio

app = FastAPI()

def send_email_notification(email: str, message: str):
    """Background task to send email"""
    print(f"Sending email to {email}: {message}")
    # Email sending logic here

def process_video_generation(video_id: str, content: str):
    """Background task for video processing"""
    print(f"Processing video {video_id} with content: {content}")
    # Video processing logic here

async def cleanup_old_files():
    """Async background task"""
    print("Cleaning up old files...")
    await asyncio.sleep(1)  # Simulate work
    print("Cleanup completed")

@app.post("/send-notification/")
async def send_notification(
    email: str,
    message: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(send_email_notification, email, message)
    return {"message": "Notification will be sent in background"}

@app.post("/generate-video/")
async def generate_video(
    video_id: str,
    content: str,
    background_tasks: BackgroundTasks
):
    # Start video generation in background
    background_tasks.add_task(process_video_generation, video_id, content)
    
    # Return immediately
    return JSONResponse(
        status_code=202,
        content={
            "message": "Video generation started",
            "video_id": video_id,
            "status": "processing"
        }
    )

@app.post("/cleanup/")
async def trigger_cleanup(background_tasks: BackgroundTasks):
    background_tasks.add_task(cleanup_old_files)
    return {"message": "Cleanup started in background"}
```

### **Q13: How do you implement WebSocket connections in FastAPI?**

**Answer:**
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
import json

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message["type"] == "chat":
                await manager.broadcast(f"Client {client_id}: {message['content']}")
            elif message["type"] == "progress":
                await manager.send_personal_message(
                    f"Progress: {message['progress']}%", 
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client {client_id} left the chat")

# Real-time progress updates
@app.websocket("/ws/progress/{task_id}")
async def progress_websocket(websocket: WebSocket, task_id: str):
    await websocket.accept()
    try:
        # Simulate progress updates
        for i in range(101):
            await websocket.send_json({
                "task_id": task_id,
                "progress": i,
                "status": "processing" if i < 100 else "completed"
            })
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
```

### **Q14: How do you implement file uploads and downloads in FastAPI?**

**Answer:**
```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import shutil
import os
from typing import List

app = FastAPI()

# Single file upload
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail="File type not allowed"
        )
    
    # Save file
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file.size,
        "path": file_path
    }

# Multiple files upload
@app.post("/upload-multiple/")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    uploaded_files = []
    
    for file in files:
        if file.filename:
            file_path = f"uploads/{file.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append({
                "filename": file.filename,
                "path": file_path
            })
    
    return {"uploaded_files": uploaded_files}

# File download
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"uploads/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

# Streaming large files
@app.get("/stream/{filename}")
async def stream_file(filename: str):
    file_path = f"uploads/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    def iterfile():
        with open(file_path, mode="rb") as file_like:
            yield from file_like
    
    return StreamingResponse(
        iterfile(), 
        media_type="application/octet-stream"
    )
```

### **Q15: How do you implement rate limiting in FastAPI?**

**Answer:**
```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict, deque
from typing import Dict

app = FastAPI()

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients: Dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old requests
        while (self.clients[client_ip] and 
               self.clients[client_ip][0] <= current_time - self.period):
            self.clients[client_ip].popleft()
        
        # Check rate limit
        if len(self.clients[client_ip]) >= self.calls:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )
        
        # Add current request
        self.clients[client_ip].append(current_time)
        
        response = await call_next(request)
        return response

# Apply rate limiting
app.add_middleware(RateLimitMiddleware, calls=100, period=60)

# Alternative: Using slowapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/data")
@limiter.limit("10/minute")
async def get_data(request: Request):
    return {"data": "some data"}

# Per-user rate limiting
def get_user_id(request: Request):
    # Extract user ID from JWT token
    return request.state.user_id

@app.post("/api/upload")
@limiter.limit("5/minute", key_func=get_user_id)
async def upload_file(request: Request):
    return {"message": "File uploaded"}
```

### **Q16: How do you implement caching in FastAPI?**

**Answer:**
```python
from fastapi import FastAPI, Depends, Request
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
import redis.asyncio as redis
from typing import Optional

app = FastAPI()

# Redis cache setup
@app.on_event("startup")
async def startup():
    redis_client = redis.from_url("redis://localhost", encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis_client), prefix="fastapi-cache")

# Simple caching
@app.get("/expensive-operation/")
@cache(expire=300)  # Cache for 5 minutes
async def expensive_operation():
    # Simulate expensive operation
    await asyncio.sleep(2)
    return {"result": "expensive data", "timestamp": time.time()}

# Conditional caching
@app.get("/user/{user_id}")
@cache(expire=600, key_builder=lambda request, **kwargs: f"user:{kwargs['user_id']}")
async def get_user(user_id: int):
    # Fetch user from database
    return {"user_id": user_id, "name": f"User {user_id}"}

# Custom cache key
def custom_key_builder(
    func,
    namespace: Optional[str] = "",
    request: Request = None,
    response=None,
    *args,
    **kwargs,
):
    prefix = FastAPICache.get_prefix()
    cache_key = f"{prefix}:{namespace}:{func.__module__}:{func.__name__}:{args}:{sorted(kwargs.items())}"
    return cache_key

@app.get("/custom-cache/")
@cache(expire=300, key_builder=custom_key_builder)
async def custom_cached_endpoint(param1: str, param2: int):
    return {"param1": param1, "param2": param2}

# Manual cache control
from fastapi_cache import FastAPICache

@app.get("/manual-cache/")
async def manual_cache():
    cache = FastAPICache.get_backend()
    
    # Check cache
    cached_data = await cache.get("manual_key")
    if cached_data:
        return cached_data
    
    # Generate data
    data = {"message": "This is cached data"}
    
    # Store in cache
    await cache.set("manual_key", data, expire=300)
    
    return data

# Cache invalidation
@app.delete("/cache/invalidate/{key}")
async def invalidate_cache(key: str):
    cache = FastAPICache.get_backend()
    await cache.delete(key)
    return {"message": f"Cache key {key} invalidated"}
```

### **Q17: How do you implement request/response logging in FastAPI?**

**Answer:**
```python
from fastapi import FastAPI, Request
from fastapi.middleware.base import BaseHTTPMiddleware
import time
import logging
import json
from typing import Callable

app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        logger.info(f"Headers: {dict(request.headers)}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(f"Response: {response.status_code}")
        logger.info(f"Process time: {process_time:.4f}s")
        
        # Add custom header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

app.add_middleware(LoggingMiddleware)

# Structured logging
import structlog

logger = structlog.get_logger()

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        
        # Extract user info if available
        user_id = getattr(request.state, 'user_id', None)
        
        # Log request
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            user_id=user_id,
            client_ip=request.client.host
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            logger.info(
                "Request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=process_time,
                user_id=user_id
            )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                "Request failed",
                method=request.method,
                url=str(request.url),
                error=str(e),
                process_time=process_time,
                user_id=user_id
            )
            raise

app.add_middleware(StructuredLoggingMiddleware)

# Custom logging endpoint
@app.get("/logs/")
async def get_logs(limit: int = 100):
    # In production, you'd read from log files or database
    return {"message": "Logs endpoint", "limit": limit}
```

### **Q18: How do you implement API versioning in FastAPI?**

**Answer:**
```python
from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse

app = FastAPI()

# Method 1: URL Path Versioning
v1_router = APIRouter(prefix="/api/v1", tags=["v1"])
v2_router = APIRouter(prefix="/api/v2", tags=["v2"])

@v1_router.get("/users/")
async def get_users_v1():
    return {"version": "v1", "users": []}

@v2_router.get("/users/")
async def get_users_v2():
    return {"version": "v2", "users": [], "features": ["enhanced_search"]}

app.include_router(v1_router)
app.include_router(v2_router)

# Method 2: Header Versioning
from fastapi import Header

@app.get("/users/")
async def get_users(api_version: str = Header(None, alias="API-Version")):
    if api_version == "v1":
        return {"version": "v1", "users": []}
    elif api_version == "v2":
        return {"version": "v2", "users": [], "features": ["enhanced_search"]}
    else:
        return {"version": "latest", "users": []}

# Method 3: Query Parameter Versioning
@app.get("/users/")
async def get_users(version: str = "latest"):
    if version == "v1":
        return {"version": "v1", "users": []}
    elif version == "v2":
        return {"version": "v2", "users": [], "features": ["enhanced_search"]}
    else:
        return {"version": "latest", "users": []}

# Method 4: Sub-application Versioning
from fastapi import FastAPI

# Create separate FastAPI instances
v1_app = FastAPI(title="API v1", version="1.0.0")
v2_app = FastAPI(title="API v2", version="2.0.0")

@v1_app.get("/users/")
async def get_users_v1():
    return {"version": "v1", "users": []}

@v2_app.get("/users/")
async def get_users_v2():
    return {"version": "v2", "users": [], "features": ["enhanced_search"]}

# Mount sub-applications
app.mount("/api/v1", v1_app)
app.mount("/api/v2", v2_app)

# Method 5: Conditional Versioning with Dependencies
from fastapi import Depends

def get_api_version(request: Request) -> str:
    # Check header first, then query param, then default
    version = request.headers.get("API-Version")
    if not version:
        version = request.query_params.get("version", "latest")
    return version

@app.get("/users/")
async def get_users(version: str = Depends(get_api_version)):
    if version == "v1":
        return {"version": "v1", "users": []}
    elif version == "v2":
        return {"version": "v2", "users": [], "features": ["enhanced_search"]}
    else:
        return {"version": "latest", "users": []}
```

### **Q19: How do you implement custom response classes in FastAPI?**

**Answer:**
```python
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Any, Dict
import json

app = FastAPI()

# Custom response class
class CustomResponse(Response):
    def __init__(self, content: Any, status_code: int = 200, **kwargs):
        super().__init__(content, status_code, **kwargs)
        self.headers["X-Custom-Header"] = "Custom Value"

# Custom JSON response
class CustomJSONResponse(JSONResponse):
    def __init__(self, content: Any, status_code: int = 200, **kwargs):
        # Add custom headers
        headers = kwargs.get("headers", {})
        headers["X-API-Version"] = "1.0"
        headers["X-Response-Time"] = str(time.time())
        kwargs["headers"] = headers
        
        super().__init__(content, status_code, **kwargs)

# Using custom responses
@app.get("/custom-response/", response_class=CustomJSONResponse)
async def get_custom_response():
    return {"message": "This is a custom response"}

# Multiple response types
@app.get("/flexible-response/")
async def get_flexible_response(
    format: str = "json",
    response: Response = None
):
    data = {"message": "Flexible response", "timestamp": time.time()}
    
    if format == "json":
        return JSONResponse(content=data)
    elif format == "html":
        html_content = f"<h1>{data['message']}</h1><p>Timestamp: {data['timestamp']}</p>"
        return HTMLResponse(content=html_content)
    elif format == "text":
        return PlainTextResponse(content=str(data))
    else:
        return data

# Response with custom headers
@app.get("/headers-response/")
async def get_headers_response():
    return JSONResponse(
        content={"message": "Response with custom headers"},
        headers={
            "X-Custom-Header": "Custom Value",
            "X-API-Version": "1.0",
            "Cache-Control": "no-cache"
        }
    )

# Streaming response
from fastapi.responses import StreamingResponse
import io

@app.get("/streaming-response/")
async def get_streaming_response():
    def generate_data():
        for i in range(10):
            yield f"Data chunk {i}\n"
            time.sleep(0.1)
    
    return StreamingResponse(
        generate_data(),
        media_type="text/plain",
        headers={"X-Streaming": "true"}
    )

# File response
from fastapi.responses import FileResponse

@app.get("/file-response/")
async def get_file_response():
    return FileResponse(
        path="path/to/file.txt",
        filename="downloaded_file.txt",
        media_type="text/plain"
    )
```

### **Q20: How do you implement custom exception handlers in FastAPI?**

**Answer:**
```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

# Custom exception classes
class CustomBusinessException(Exception):
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class InsufficientPermissionsException(Exception):
    def __init__(self, resource: str, action: str):
        self.resource = resource
        self.action = action
        self.message = f"Insufficient permissions for {action} on {resource}"
        super().__init__(self.message)

# Global exception handler
@app.exception_handler(CustomBusinessException)
async def custom_business_exception_handler(request: Request, exc: CustomBusinessException):
    logger.error(f"Business exception: {exc.message}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Business Logic Error",
            "message": exc.message,
            "error_code": exc.error_code,
            "path": str(request.url)
        }
    )

@app.exception_handler(InsufficientPermissionsException)
async def insufficient_permissions_handler(request: Request, exc: InsufficientPermissionsException):
    logger.warning(f"Permission denied: {exc.message}")
    return JSONResponse(
        status_code=403,
        content={
            "error": "Insufficient Permissions",
            "message": exc.message,
            "resource": exc.resource,
            "action": exc.action,
            "path": str(request.url)
        }
    )

# Validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Request validation failed",
            "details": exc.errors(),
            "path": str(request.url)
        }
    )

# Generic exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "path": str(request.url)
        }
    )

# Using custom exceptions
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    if user_id < 1:
        raise CustomBusinessException(
            "User ID must be positive",
            error_code="INVALID_USER_ID"
        )
    
    # Simulate permission check
    if user_id > 1000:
        raise InsufficientPermissionsException("user", "read")
    
    return {"user_id": user_id, "name": f"User {user_id}"}

# Error response model
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    error: str
    message: str
    error_code: str = None
    details: dict = None
    path: str = None
    timestamp: str = None

# Structured error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP Error",
            message=exc.detail,
            path=str(request.url),
            timestamp=datetime.now().isoformat()
        ).dict()
    )
```

---

## **3. DATABASE & ORM**

### **Q21: How do you set up SQLAlchemy with FastAPI for async operations?**

**Answer:**
Based on the NeuroShiksha database setup:

```python
# app/database.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import MetaData
from .config import settings

# Create declarative base
Base = declarative_base()

# Global variables for lazy initialization
engine = None
_AsyncSessionLocal = None

def get_engine():
    """Lazy initialization of database engine"""
    global engine
    if engine is None:
        engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=20,
            max_overflow=30,
            pool_timeout=30,
        )
    return engine

def _get_async_session_local():
    """Get or create the async session local"""
    global _AsyncSessionLocal
    if _AsyncSessionLocal is None:
        _AsyncSessionLocal = sessionmaker(
            get_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _AsyncSessionLocal

# Dependency to get database session
async def get_db():
    session_factory = _get_async_session_local()
    async with session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Initialize database
async def init_db():
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Health check
async def check_db_health():
    try:
        session_factory = _get_async_session_local()
        async with session_factory() as session:
            result = await session.execute(text("SELECT 1"))
            row = result.fetchone()
            return row is not None and row[0] == 1
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return False
```

### **Q22: How do you create and use SQLAlchemy models with relationships?**

**Answer:**
```python
# app/models/user.py
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..database import Base
import uuid

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    full_name = Column(String(255))
    hashed_password = Column(String(255), nullable=False)
    learning_preference = Column(String(50), default="default")
    role = Column(String(50), default="student")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    courses = relationship("UserCourseEnrollment", back_populates="user")
    progress = relationship("UserProgress", back_populates="user")

# app/models/course.py
class Course(Base):
    __tablename__ = "courses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    difficulty_level = Column(Integer, default=1)
    creator_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    creator = relationship("User", back_populates="created_courses")
    enrollments = relationship("UserCourseEnrollment", back_populates="course")
    lessons = relationship("Lesson", back_populates="course", cascade="all, delete-orphan")

# app/models/enrollment.py
class UserCourseEnrollment(Base):
    __tablename__ = "user_course_enrollments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    course_id = Column(UUID(as_uuid=True), ForeignKey("courses.id"), nullable=False)
    enrolled_at = Column(DateTime(timezone=True), server_default=func.now())
    progress_percentage = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="courses")
    course = relationship("Course", back_populates="enrollments")

# Using relationships in services
class CourseService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_course_with_enrollments(self, course_id: str):
        result = await self.db.execute(
            select(Course)
            .options(joinedload(Course.enrollments).joinedload(UserCourseEnrollment.user))
            .where(Course.id == course_id)
        )
        return result.scalar_one_or_none()

    async def get_user_courses(self, user_id: str):
        result = await self.db.execute(
            select(Course)
            .join(UserCourseEnrollment)
            .where(UserCourseEnrollment.user_id == user_id)
        )
        return result.scalars().all()
```

### **Q23: How do you implement database migrations with Alembic?**

**Answer:**
```python
# alembic.ini
[alembic]
script_location = alembic
prepend_sys_path = .
version_path_separator = os
sqlalch
## **Q24-Q50: DATABASE & ORM CONTINUED**

### **Q24: How do you handle database transactions in FastAPI?**
```python
async def create_user_with_profile(user_data: UserCreate, db: AsyncSession):
    async with db.begin():  # Start transaction
        user = User(**user_data.dict())
        db.add(user)
        await db.flush()  # Get user ID
        
        profile = UserProfile(user_id=user.id, **profile_data.dict())
        db.add(profile)
        # Transaction commits automatically on success
```

### **Q25: What's the difference between flush() and commit()?**
- `flush()`: Sends SQL to database but doesn't commit transaction
- `commit()`: Commits the transaction, making changes permanent
- Use `flush()` when you need the generated ID before committing

### **Q26: How do you handle database connection pooling?**
```python
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,        # Base connections
    max_overflow=30,     # Additional connections
    pool_pre_ping=True,  # Validate connections
    pool_recycle=300     # Recycle connections every 5 minutes
)
```

### **Q27: How do you implement soft deletes?**
```python
class User(Base):
    __tablename__ = "users"
    id = Column(UUID, primary_key=True)
    email = Column(String, nullable=False)
    deleted_at = Column(DateTime, nullable=True)  # Soft delete flag
    
    @property
    def is_deleted(self):
        return self.deleted_at is not None
```

### **Q28: How do you handle database indexes?**
```python
class User(Base):
    __tablename__ = "users"
    id = Column(UUID, primary_key=True)
    email = Column(String, nullable=False, index=True)  # Single column index
    created_at = Column(DateTime, index=True)
    
    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),  # Composite index
        Index('idx_user_created_at', 'created_at', postgresql_using='btree'),
    )
```

### **Q29: How do you implement database pagination?**
```python
async def get_users_paginated(db: AsyncSession, page: int = 1, size: int = 10):
    offset = (page - 1) * size
    result = await db.execute(
        select(User)
        .offset(offset)
        .limit(size)
        .order_by(User.created_at.desc())
    )
    return result.scalars().all()
```

### **Q30: How do you handle database constraints?**
```python
class User(Base):
    __tablename__ = "users"
    id = Column(UUID, primary_key=True)
    email = Column(String, nullable=False, unique=True)  # Unique constraint
    age = Column(Integer, CheckConstraint('age >= 18'))  # Check constraint
    
    __table_args__ = (
        UniqueConstraint('email', 'phone', name='unique_email_phone'),
        ForeignKeyConstraint(['department_id'], ['departments.id']),
    )
```

## **Q31-Q60: AUTHENTICATION & SECURITY**

### **Q31: How do you implement JWT authentication?**
```python
from jose import JWTError, jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None
```

### **Q32: How do you hash passwords securely?**
```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
```

### **Q33: How do you implement role-based access control?**
```python
from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    TEACHER = "teacher"
    STUDENT = "student"

def require_role(required_role: UserRole):
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role != required_role:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return current_user
    return role_checker

@app.get("/admin-only/")
async def admin_endpoint(user: User = Depends(require_role(UserRole.ADMIN))):
    return {"message": "Admin access granted"}
```

### **Q34: How do you implement OAuth2 with FastAPI?**
```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials"
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = await get_user(username)
    if user is None:
        raise credentials_exception
    return user
```

### **Q35: How do you implement API key authentication?**
```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != "your-secret-api-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.get("/protected/")
async def protected_endpoint(api_key: str = Depends(verify_api_key)):
    return {"message": "Access granted"}
```

### **Q36: How do you implement session-based authentication?**
```python
from fastapi import Request
import secrets

# Store sessions in memory (use Redis in production)
sessions = {}

def create_session(user_id: str) -> str:
    session_id = secrets.token_urlsafe(32)
    sessions[session_id] = {"user_id": user_id, "created_at": datetime.now()}
    return session_id

async def get_current_user_from_session(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return sessions[session_id]["user_id"]
```

### **Q37: How do you implement password reset functionality?**
```python
import secrets
from datetime import timedelta

reset_tokens = {}  # Use Redis in production

def generate_reset_token() -> str:
    return secrets.token_urlsafe(32)

async def send_password_reset(email: str):
    token = generate_reset_token()
    reset_tokens[token] = {
        "email": email,
        "expires": datetime.now() + timedelta(hours=1)
    }
    # Send email with reset link
    return token

async def reset_password(token: str, new_password: str):
    if token not in reset_tokens:
        raise HTTPException(status_code=400, detail="Invalid token")
    
    token_data = reset_tokens[token]
    if datetime.now() > token_data["expires"]:
        del reset_tokens[token]
        raise HTTPException(status_code=400, detail="Token expired")
    
    # Update password
    await update_user_password(token_data["email"], new_password)
    del reset_tokens[token]
```

### **Q38: How do you implement two-factor authentication?**
```python
import pyotp
import qrcode

def generate_2fa_secret(user_id: str) -> str:
    secret = pyotp.random_base32()
    # Store secret in database
    return secret

def generate_2fa_qr(user_id: str, secret: str) -> str:
    totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
        name=user_id,
        issuer_name="YourApp"
    )
    return totp_uri

def verify_2fa_token(secret: str, token: str) -> bool:
    totp = pyotp.TOTP(secret)
    return totp.verify(token, valid_window=1)
```

### **Q39: How do you implement rate limiting per user?**
```python
from collections import defaultdict
import time

user_requests = defaultdict(list)

def rate_limit_per_user(user_id: str, max_requests: int = 100, window: int = 3600):
    current_time = time.time()
    user_requests[user_id] = [
        req_time for req_time in user_requests[user_id]
        if current_time - req_time < window
    ]
    
    if len(user_requests[user_id]) >= max_requests:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    user_requests[user_id].append(current_time)
```

### **Q40: How do you implement CORS properly?**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

## **Q41-Q70: ASYNC PROGRAMMING**

### **Q41: What's the difference between async and sync in FastAPI?**
```python
# Sync - blocks the event loop
@app.get("/sync/")
def sync_endpoint():
    time.sleep(1)  # Blocks
    return {"message": "sync"}

# Async - non-blocking
@app.get("/async/")
async def async_endpoint():
    await asyncio.sleep(1)  # Non-blocking
    return {"message": "async"}
```

### **Q42: How do you handle async database operations?**
```python
async def get_user_async(db: AsyncSession, user_id: str):
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()

async def create_user_async(db: AsyncSession, user_data: UserCreate):
    user = User(**user_data.dict())
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user
```

### **Q43: How do you run multiple async operations concurrently?**
```python
import asyncio

async def fetch_user_data(user_id: str):
    # Run multiple operations concurrently
    user, courses, progress = await asyncio.gather(
        get_user(user_id),
        get_user_courses(user_id),
        get_user_progress(user_id)
    )
    return {"user": user, "courses": courses, "progress": progress}
```

### **Q44: How do you handle async context managers?**
```python
class AsyncDatabaseManager:
    async def __aenter__(self):
        self.session = await create_session()
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.session.rollback()
        await self.session.close()

# Usage
async def some_operation():
    async with AsyncDatabaseManager() as db:
        # Database operations
        pass
```

### **Q45: How do you implement async generators?**
```python
async def stream_large_dataset():
    for i in range(1000):
        data = await fetch_data_chunk(i)
        yield data
        await asyncio.sleep(0.1)  # Prevent overwhelming

@app.get("/stream/")
async def stream_endpoint():
    return StreamingResponse(stream_large_dataset())
```

### **Q46: How do you handle async exceptions?**
```python
async def risky_async_operation():
    try:
        result = await external_api_call()
        return result
    except asyncio.TimeoutError:
        logger.error("API call timed out")
        raise HTTPException(status_code=504, detail="External service timeout")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### **Q47: How do you implement async locks?**
```python
import asyncio

# Global lock for shared resource
resource_lock = asyncio.Lock()

async def update_shared_resource():
    async with resource_lock:
        # Critical section - only one coroutine at a time
        await modify_shared_data()
```

### **Q48: How do you handle async timeouts?**
```python
async def api_call_with_timeout():
    try:
        result = await asyncio.wait_for(
            external_api_call(),
            timeout=30.0  # 30 seconds timeout
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
```

### **Q49: How do you implement async queues?**
```python
import asyncio

# Global queue for background tasks
task_queue = asyncio.Queue()

async def background_worker():
    while True:
        task = await task_queue.get()
        await process_task(task)
        task_queue.task_done()

async def add_background_task(task_data):
    await task_queue.put(task_data)
```

### **Q50: How do you handle async cleanup?**
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await initialize_resources()
    yield
    # Shutdown
    await cleanup_resources()

app = FastAPI(lifespan=lifespan)
```

## **Q51-Q80: TESTING & QUALITY ASSURANCE**

### **Q51: How do you write unit tests for FastAPI?**
```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}

def test_create_user():
    response = client.post("/users/", json={"email": "test@example.com"})
    assert response.status_code == 201
    assert "id" in response.json()
```

### **Q52: How do you test async endpoints?**
```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_async_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/async-endpoint/")
        assert response.status_code == 200
```

### **Q53: How do you mock external services in tests?**
```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_with_mocked_service():
    with patch('app.services.external_api') as mock_api:
        mock_api.return_value = AsyncMock(return_value={"data": "mocked"})
        
        response = await client.get("/endpoint-using-external-api/")
        assert response.status_code == 200
```

### **Q54: How do you test database operations?**
```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import get_db, Base

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture
def test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
```

### **Q55: How do you test authentication?**
```python
def test_protected_endpoint():
    # Test without token
    response = client.get("/protected/")
    assert response.status_code == 401
    
    # Test with valid token
    token = create_test_token()
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/protected/", headers=headers)
    assert response.status_code == 200
```

### **Q56: How do you test error handling?**
```python
def test_error_handling():
    response = client.get("/users/999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
```

### **Q57: How do you test file uploads?**
```python
def test_file_upload():
    with open("test_file.txt", "rb") as f:
        response = client.post("/upload/", files={"file": f})
    assert response.status_code == 200
    assert "filename" in response.json()
```

### **Q58: How do you test WebSocket connections?**
```python
def test_websocket():
    with client.websocket_connect("/ws/123") as websocket:
        data = websocket.receive_json()
        assert data == {"message": "Connected"}
        
        websocket.send_json({"type": "chat", "content": "Hello"})
        data = websocket.receive_json()
        assert "Hello" in data["message"]
```

### **Q59: How do you test background tasks?**
```python
def test_background_task():
    with patch('app.tasks.send_email') as mock_send:
        response = client.post("/send-notification/", json={"email": "test@example.com"})
        assert response.status_code == 200
        
        # Background task should be called
        mock_send.assert_called_once()
```

### **Q60: How do you test rate limiting?**
```python
def test_rate_limiting():
    # Make requests up to the limit
    for _ in range(100):
        response = client.get("/rate-limited/")
        assert response.status_code == 200
    
    # Next request should be rate limited
    response = client.get("/rate-limited/")
    assert response.status_code == 429
```

## **Q61-Q90: PERFORMANCE & OPTIMIZATION**

### **Q61: How do you optimize database queries?**
```python
# Use select_related for foreign keys
result = await db.execute(
    select(User)
    .options(joinedload(User.courses))
    .where(User.id == user_id)
)

# Use only() to select specific fields
result = await db.execute(
    select(User.id, User.email)
    .where(User.active == True)
)
```

### **Q62: How do you implement connection pooling?**
```python
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,           # Base connections
    max_overflow=30,        # Additional connections
    pool_pre_ping=True,     # Validate connections
    pool_recycle=3600,      # Recycle every hour
    echo=False              # Disable SQL logging in production
)
```

### **Q63: How do you implement response caching?**
```python
from functools import lru_cache
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

@lru_cache(maxsize=128)
def get_cached_data(key: str):
    return expensive_computation(key)

async def get_data_with_redis_cache(key: str):
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    
    data = await expensive_async_operation(key)
    redis_client.setex(key, 3600, json.dumps(data))  # Cache for 1 hour
    return data
```

### **Q64: How do you optimize JSON serialization?**
```python
from pydantic import BaseModel
from typing import Any
import orjson

class OptimizedResponse(BaseModel):
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
        json_loads = orjson.loads
        json_dumps = orjson.dumps
```

### **Q65: How do you implement lazy loading?**
```python
class User(Base):
    __tablename__ = "users"
    id = Column(UUID, primary_key=True)
    email = Column(String)
    
    # Lazy relationship
    courses = relationship("Course", lazy="select")
    
    # Eager loading when needed
    @classmethod
    async def get_with_courses(cls, db: AsyncSession, user_id: str):
        result = await db.execute(
            select(cls)
            .options(joinedload(cls.courses))
            .where(cls.id == user_id)
        )
        return result.scalar_one_or_none()
```

### **Q66: How do you implement pagination efficiently?**
```python
async def get_paginated_results(
    db: AsyncSession, 
    model: Type[Base], 
    page: int = 1, 
    size: int = 20
):
    offset = (page - 1) * size
    
    # Get total count
    count_result = await db.execute(select(func.count(model.id)))
    total = count_result.scalar()
    
    # Get paginated results
    result = await db.execute(
        select(model)
        .offset(offset)
        .limit(size)
        .order_by(model.created_at.desc())
    )
    
    return {
        "items": result.scalars().all(),
        "total": total,
        "page": page,
        "size": size,
        "pages": (total + size - 1) // size
    }
```

### **Q67: How do you implement database indexing?**
```python
class User(Base):
    __tablename__ = "users"
    id = Column(UUID, primary_key=True)
    email = Column(String, index=True)  # Single column index
    created_at = Column(DateTime, index=True)
    
    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),  # Composite
        Index('idx_user_created_at', 'created_at', postgresql_using='btree'),
    )
```

### **Q68: How do you implement query optimization?**
```python
# Use exists() instead of count() for existence checks
async def user_exists(db: AsyncSession, email: str) -> bool:
    result = await db.execute(
        select(exists().where(User.email == email))
    )
    return result.scalar()

# Use bulk operations
async def bulk_create_users(db: AsyncSession, users_data: List[dict]):
    users = [User(**data) for data in users_data]
    db.add_all(users)
    await db.commit()
```

### **Q69: How do you implement memory optimization?**
```python
# Use generators for large datasets
async def stream_large_dataset():
    async for chunk in get_data_chunks():
        yield chunk

# Use __slots__ for memory efficiency
class OptimizedModel(BaseModel):
    __slots__ = ['id', 'name', 'email']
    
    id: str
    name: str
    email: str
```

### **Q70: How do you implement async optimization?**
```python
# Use asyncio.gather for concurrent operations
async def get_user_dashboard(user_id: str):
    user, courses, progress = await asyncio.gather(
        get_user(user_id),
        get_user_courses(user_id),
        get_user_progress(user_id),
        return_exceptions=True
    )
    return {"user": user, "courses": courses, "progress": progress}
```

## **Q71-Q100: ARCHITECTURE & DESIGN PATTERNS**

### **Q71: How do you implement the Repository pattern?**
```python
class UserRepository:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create(self, user_data: UserCreate) -> User:
        user = User(**user_data.dict())
        self.db.add(user)
        await self.db.commit()
        return user
    
    async def get_by_id(self, user_id: str) -> Optional[User]:
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
    
    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()
```

### **Q72: How do you implement the Service layer pattern?**
```python
class UserService:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo
    
    async def create_user(self, user_data: UserCreate) -> User:
        # Business logic
        if await self.user_repo.get_by_email(user_data.email):
            raise ValueError("Email already exists")
        
        # Hash password
        hashed_password = hash_password(user_data.password)
        user_data.hashed_password = hashed_password
        
        return await self.user_repo.create(user_data)
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        user = await self.user_repo.get_by_email(email)
        if user and verify_password(password, user.hashed_password):
            return user
        return None
```

### **Q73: How do you implement the Factory pattern?**
```python
class NotificationFactory:
    @staticmethod
    def create_notifier(notification_type: str):
        if notification_type == "email":
            return EmailNotifier()
        elif notification_type == "sms":
            return SMSNotifier()
        elif notification_type == "push":
            return PushNotifier()
        else:
            raise ValueError(f"Unknown notification type: {notification_type}")

# Usage
notifier = NotificationFactory.create_notifier("email")
await notifier.send("Hello", "user@example.com")
```

### **Q74: How do you implement the Observer pattern?**
```python
from typing import List, Callable
import asyncio

class EventBus:
    def __init__(self):
        self.subscribers: List[Callable] = []
    
    def subscribe(self, callback: Callable):
        self.subscribers.append(callback)
    
    async def publish(self, event: dict):
        tasks = [callback(event) for callback in self.subscribers]
        await asyncio.gather(*tasks, return_exceptions=True)

# Usage
event_bus = EventBus()

@event_bus.subscribe
async def on_user_created(event):
    print(f"User created: {event['user_id']}")

await event_bus.publish({"type": "user_created", "user_id": "123"})
```

### **Q75: How do you implement the Strategy pattern?**
```python
from abc import ABC, abstractmethod

class PaymentStrategy(ABC):
    @abstractmethod
    async def process_payment(self, amount: float) -> bool:
        pass

class CreditCardPayment(PaymentStrategy):
    async def process_payment(self, amount: float) -> bool:
        # Credit card logic
        return True

class PayPalPayment(PaymentStrategy):
    async def process_payment(self, amount: float) -> bool:
        # PayPal logic
        return True

class PaymentProcessor:
    def __init__(self, strategy: PaymentStrategy):
        self.strategy = strategy
    
    async def process(self, amount: float) -> bool:
        return await self.strategy.process_payment(amount)
```

### **Q76: How do you implement the Command pattern?**
```python
from abc import ABC, abstractmethod

class Command(ABC):
    @abstractmethod
    async def execute(self):
        pass
    
    @abstractmethod
    async def undo(self):
        pass

class CreateUserCommand(Command):
    def __init__(self, user_service: UserService, user_data: UserCreate):
        self.user_service = user_service
        self.user_data = user_data
        self.created_user = None
    
    async def execute(self):
        self.created_user = await self.user_service.create_user(self.user_data)
        return self.created_user
    
    async def undo(self):
        if self.created_user:
            await self.user_service.delete_user(self.created_user.id)

class CommandInvoker:
    def __init__(self):
        self.history: List[Command] = []
    
    async def execute_command(self, command: Command):
        result = await command.execute()
        self.history.append(command)
        return result
    
    async def undo_last(self):
        if self.history:
            command = self.history.pop()
            await command.undo()
```

### **Q77: How do you implement the Decorator pattern?**
```python
def retry(max_attempts: int = 3):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            return None
        return wrapper
    return decorator

@retry(max_attempts=3)
async def unreliable_api_call():
    # API call that might fail
    pass
```

### **Q78: How do you implement the Singleton pattern?**
```python
class DatabaseManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.connection = None
            self._initialized = True
    
    async def connect(self):
        if not self.connection:
            self.connection = await create_connection()
        return self.connection
```

### **Q79: How do you implement the Adapter pattern?**
```python
class LegacyAPI:
    def get_user_info(self, user_id: int):
        return {"id": user_id, "name": f"User {user_id}"}

class ModernUserAPI:
    def __init__(self, legacy_api: LegacyAPI):
        self.legacy_api = legacy_api
    
    async def get_user(self, user_id: str) -> User:
        # Adapt legacy API to modern interface
        legacy_data = self.legacy_api.get_user_info(int(user_id))
        return User(
            id=user_id,
            name=legacy_data["name"],
            email=f"user{user_id}@example.com"
        )
```

### **Q80: How do you implement the Facade pattern?**
```python
class UserFacade:
    def __init__(self, user_service: UserService, auth_service: AuthService, notification_service: NotificationService):
        self.user_service = user_service
        self.auth_service = auth_service
        self.notification_service = notification_service
    
    async def register_user(self, user_data: UserCreate) -> dict:
        # Complex registration process simplified
        user = await self.user_service.create_user(user_data)
        token = await self.auth_service.generate_token(user.id)
        await self.notification_service.send_welcome_email(user.email)
        
        return {
            "user": user,
            "token": token,
            "message": "Registration successful"
        }
```

## **Q81-Q110: DEVOPS & DEPLOYMENT**

### **Q81: How do you containerize a FastAPI application?**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **Q82: How do you use Docker Compose for development?**
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/dbname
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: dbname
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### **Q83: How do you implement health checks?**
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health/detailed")
async def detailed_health_check(db: AsyncSession = Depends(get_db)):
    checks = {
        "database": await check_db_health(db),
        "redis": await check_redis_health(),
        "external_api": await check_external_api()
    }
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_healthy else "unhealthy",
            "checks": checks
        }
    )
```

### **Q84: How do you implement logging in production?**
```python
import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host
    )
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    
    return response
```

### **Q85: How do you implement environment configuration?**
```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    database_url: str
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # External APIs
    openai_api_key: Optional[str] = None
    redis_url: str = "redis://localhost:6379"
    
    # App settings
    debug: bool = False
    allowed_origins: list = ["http://localhost:3000"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### **Q86: How do you implement graceful shutdown?**
```python
from contextlib import asynccontextmanager
import signal
import asyncio

shutdown_event = asyncio.Event()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_tasks()
    
    # Setup signal handlers
    def signal_handler():
        shutdown_event.set()
    
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGTERM, signal_handler)
    loop.add_signal_handler(signal.SIGINT, signal_handler)
    
    yield
    
    # Shutdown
    await shutdown_tasks()

app = FastAPI(lifespan=lifespan)
```

### **Q87: How do you implement monitoring and metrics?**
```python
from prometheus_client import Counter, Histogram, generate_latest
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### **Q88: How do you implement database migrations?**
```python
# alembic/env.py
from alembic import context
from app.database import Base
from app.models import *  # Import all models

target_metadata = Base.metadata

# alembic.ini
[alembic]
script_location = alembic
sqlalchemy.url = postgresql://user:pass@localhost/dbname

# Commands
# alembic revision --autogenerate -m "Add user table"
# alembic upgrade head
# alembic downgrade -1
```

### **Q89: How do you implement CI/CD pipeline?**
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio
    
    - name: Run tests
      run: pytest
    
    - name: Build Docker image
      run: docker build -t myapp:${{ github.sha }} .
    
    - name: Deploy to production
      if: github.ref == 'refs/heads/main'
      run: |
        # Deployment commands
        echo "Deploying to production..."
```

### **Q90: How do you implement load balancing?**
```python
# Using nginx configuration
upstream fastapi_backend {
    server app1:8000;
    server app2:8000;
    server app3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://fastapi_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## **Q91-Q120: AI/ML INTEGRATION**

### **Q91: How do you integrate OpenAI with FastAPI?**
```python
import openai
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=settings.openai_api_key)

async def generate_content(prompt: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response.choices[0].message.content

@app.post("/generate/")
async def generate_text(request: GenerateRequest):
    content = await generate_content(request.prompt)
    return {"content": content}
```

### **Q92: How do you implement vector embeddings?**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

async def create_embedding(text: str) -> List[float]:
    embedding = model.encode(text)
    return embedding.tolist()

async def find_similar_texts(query: str, texts: List[str], threshold: float = 0.8):
    query_embedding = await create_embedding(query)
    text_embeddings = [await create_embedding(text) for text in texts]
    
    similarities = []
    for i, text_embedding in enumerate(text_embeddings):
        similarity = np.dot(query_embedding, text_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
        )
        if similarity > threshold:
            similarities.append((texts[i], similarity))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)
```

### **Q93: How do you implement RAG (Retrieval Augmented Generation)?**
```python
class RAGService:
    def __init__(self, vector_store, llm_client):
        self.vector_store = vector_store
        self.llm_client = llm_client
    
    async def query(self, question: str) -> str:
        # Retrieve relevant documents
        relevant_docs = await self.vector_store.similarity_search(question, k=5)
        
        # Create context
        context = "\n".join([doc.content for doc in relevant_docs])
        
        # Generate answer
        prompt = f"""
        Context: {context}
        Question: {question}
        Answer:
        """
        
        response = await self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
```

### **Q94: How do you implement streaming responses for AI?**
```python
from fastapi.responses import StreamingResponse

async def stream_ai_response(prompt: str):
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

@app.post("/stream-chat/")
async def stream_chat(request: ChatRequest):
    return StreamingResponse(
        stream_ai_response(request.prompt),
        media_type="text/plain"
    )
```

### **Q95: How do you implement AI model caching?**
```python
import hashlib
from functools import lru_cache

class AIModelCache:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def get_cached_response(self, prompt: str) -> Optional[str]:
        cache_key = f"ai_response:{hashlib.md5(prompt.encode()).hexdigest()}"
        return await self.redis.get(cache_key)
    
    async def cache_response(self, prompt: str, response: str, ttl: int = 3600):
        cache_key = f"ai_response:{hashlib.md5(prompt.encode()).hexdigest()}"
        await self.redis.setex(cache_key, ttl, response)
    
    async def get_or_generate(self, prompt: str, generator_func):
        cached = await self.get_cached_response(prompt)
        if cached:
            return cached
        
        response = await generator_func(prompt)
        await self.cache_response(prompt, response)
        return response
```

### **Q96: How do you implement AI model versioning?**
```python
class ModelManager:
    def __init__(self):
        self.models = {}
        self.default_model = "gpt-3.5-turbo"
    
    def register_model(self, name: str, model_config: dict):
        self.models[name] = model_config
    
    async def get_model_response(self, model_name: str, prompt: str):
        if model_name not in self.models:
            model_name = self.default_model
        
        model_config = self.models[model_name]
        
        response = await client.chat.completions.create(
            model=model_config["model"],
            messages=[{"role": "user", "content": prompt}],
            **model_config.get("params", {})
        )
        
        retur
### **Q97: How do you implement AI model fallback strategies?**
```python
class AIFallbackService:
    def __init__(self, primary_model, fallback_models):
        self.primary_model = primary_model
        self.fallback_models = fallback_models
    
    async def generate_with_fallback(self, prompt: str):
        try:
            return await self.primary_model.generate(prompt)
        except Exception as e:
            logger.warning(f"Primary model failed: {e}")
            
            for fallback in self.fallback_models:
                try:
                    return await fallback.generate(prompt)
                except Exception as e:
                    logger.warning(f"Fallback model failed: {e}")
                    continue
            
            raise Exception("All models failed")
```

### **Q98: How do you implement AI response validation?**
```python
from pydantic import BaseModel, validator

class AIResponse(BaseModel):
    content: str
    confidence: float
    model_used: str
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v
    
    @validator('content')
    def validate_content(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Content too short')
        return v

async def validate_ai_response(response: str) -> AIResponse:
    # Implement validation logic
    confidence = calculate_confidence(response)
    return AIResponse(
        content=response,
        confidence=confidence,
        model_used="gpt-3.5-turbo"
    )
```

### **Q99: How do you implement AI prompt engineering patterns?**
```python
class PromptTemplate:
    def __init__(self, template: str):
        self.template = template
    
    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

class PromptManager:
    def __init__(self):
        self.templates = {
            "educational": PromptTemplate(
                "You are an educational AI assistant. Create a lesson about {topic} "
                "for {level} students. Include examples and exercises."
            ),
            "creative": PromptTemplate(
                "You are a creative writer. Write a {genre} story about {theme} "
                "with {characters} characters."
            )
        }
    
    def get_prompt(self, template_name: str, **kwargs) -> str:
        if template_name not in self.templates:
            raise ValueError(f"Template {template_name} not found")
        return self.templates[template_name].format(**kwargs)
```

### **Q100: How do you implement AI cost optimization?**
```python
class AICostOptimizer:
    def __init__(self):
        self.model_costs = {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "claude-3": 0.015
        }
        self.usage_tracker = {}
    
    async def select_optimal_model(self, task_complexity: str, budget: float):
        if task_complexity == "simple" and budget < 0.01:
            return "gpt-3.5-turbo"
        elif task_complexity == "complex":
            return "gpt-4"
        else:
            return "claude-3"
    
    def track_usage(self, model: str, tokens: int):
        cost = self.model_costs[model] * (tokens / 1000)
        self.usage_tracker[model] = self.usage_tracker.get(model, 0) + cost
```

## **Q101-Q130: SYSTEM DESIGN & SCALABILITY**

### **Q101: How do you implement microservices communication?**
```python
import httpx
from typing import Dict, Any

class ServiceClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
    
    async def call_service(self, endpoint: str, method: str = "GET", data: Dict = None):
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = f"{self.base_url}{endpoint}"
            
            if method == "GET":
                response = await client.get(url)
            elif method == "POST":
                response = await client.post(url, json=data)
            
            response.raise_for_status()
            return response.json()

# Usage
user_service = ServiceClient("http://user-service:8001")
course_service = ServiceClient("http://course-service:8002")

async def get_user_with_courses(user_id: str):
    user = await user_service.call_service(f"/users/{user_id}")
    courses = await course_service.call_service(f"/users/{user_id}/courses")
    return {"user": user, "courses": courses}
```

### **Q102: How do you implement circuit breaker pattern?**
```python
import asyncio
from enum import Enum
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        return (time.time() - self.last_failure_time) > self.timeout
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

### **Q103: How do you implement bulkhead pattern?**
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BulkheadExecutor:
    def __init__(self):
        self.cpu_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cpu")
        self.io_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="io")
        self.ai_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ai")
    
    async def execute_cpu_task(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.cpu_pool, func, *args)
    
    async def execute_io_task(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.io_pool, func, *args)
    
    async def execute_ai_task(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.ai_pool, func, *args)

# Usage
bulkhead = BulkheadExecutor()

async def process_user_data(user_id: str):
    # CPU-intensive task
    processed_data = await bulkhead.execute_cpu_task(heavy_computation, user_id)
    
    # I/O task
    saved_data = await bulkhead.execute_io_task(save_to_database, processed_data)
    
    # AI task
    ai_result = await bulkhead.execute_ai_task(generate_insights, saved_data)
    
    return ai_result
```

### **Q104: How do you implement event sourcing?**
```python
from typing import List, Dict, Any
from datetime import datetime
import json

class Event:
    def __init__(self, event_type: str, data: Dict[str, Any], aggregate_id: str):
        self.event_type = event_type
        self.data = data
        self.aggregate_id = aggregate_id
        self.timestamp = datetime.now()
        self.version = 1

class EventStore:
    def __init__(self):
        self.events: List[Event] = []
    
    async def append_event(self, event: Event):
        self.events.append(event)
        # In production, save to database
    
    async def get_events(self, aggregate_id: str) -> List[Event]:
        return [e for e in self.events if e.aggregate_id == aggregate_id]

class UserAggregate:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.name = None
        self.email = None
        self.version = 0
    
    def apply_event(self, event: Event):
        if event.event_type == "UserCreated":
            self.name = event.data["name"]
            self.email = event.data["email"]
        elif event.event_type == "UserUpdated":
            if "name" in event.data:
                self.name = event.data["name"]
            if "email" in event.data:
                self.email = event.data["email"]
        
        self.version += 1
    
    @classmethod
    def from_events(cls, user_id: str, events: List[Event]):
        aggregate = cls(user_id)
        for event in events:
            aggregate.apply_event(event)
        return aggregate
```

### **Q105: How do you implement CQRS (Command Query Responsibility Segregation)?**
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

# Commands (Write operations)
class Command(ABC):
    pass

class CreateUserCommand(Command):
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

class UpdateUserCommand(Command):
    def __init__(self, user_id: str, name: str = None, email: str = None):
        self.user_id = user_id
        self.name = name
        self.email = email

# Queries (Read operations)
class Query(ABC):
    pass

class GetUserQuery(Query):
    def __init__(self, user_id: str):
        self.user_id = user_id

class ListUsersQuery(Query):
    def __init__(self, limit: int = 10, offset: int = 0):
        self.limit = limit
        self.offset = offset

# Command Handler
class CommandHandler:
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    async def handle_create_user(self, command: CreateUserCommand):
        event = Event(
            event_type="UserCreated",
            data={"name": command.name, "email": command.email},
            aggregate_id=str(uuid.uuid4())
        )
        await self.event_store.append_event(event)
        return event.aggregate_id
    
    async def handle_update_user(self, command: UpdateUserCommand):
        event = Event(
            event_type="UserUpdated",
            data={"name": command.name, "email": command.email},
            aggregate_id=command.user_id
        )
        await self.event_store.append_event(event)

# Query Handler
class QueryHandler:
    def __init__(self, read_model):
        self.read_model = read_model
    
    async def handle_get_user(self, query: GetUserQuery):
        return await self.read_model.get_user(query.user_id)
    
    async def handle_list_users(self, query: ListUsersQuery):
        return await self.read_model.list_users(query.limit, query.offset)
```

### **Q106: How do you implement distributed caching?**
```python
import redis
import json
from typing import Any, Optional
import hashlib

class DistributedCache:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def _generate_key(self, prefix: str, *args) -> str:
        key_data = ":".join(str(arg) for arg in args)
        hash_key = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{hash_key}"
    
    async def get(self, prefix: str, *args) -> Optional[Any]:
        key = self._generate_key(prefix, *args)
        data = await self.redis.get(key)
        return json.loads(data) if data else None
    
    async def set(self, prefix: str, value: Any, ttl: int = 3600, *args):
        key = self._generate_key(prefix, *args)
        await self.redis.setex(key, ttl, json.dumps(value))
    
    async def delete(self, prefix: str, *args):
        key = self._generate_key(prefix, *args)
        await self.redis.delete(key)
    
    async def invalidate_pattern(self, pattern: str):
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)

# Usage
cache = DistributedCache(redis_client)

async def get_user_cached(user_id: str):
    cached = await cache.get("user", user_id)
    if cached:
        return cached
    
    user = await get_user_from_db(user_id)
    await cache.set("user", user, 3600, user_id)
    return user
```

### **Q107: How do you implement message queues?**
```python
import asyncio
from typing import Dict, Any, Callable
import json

class MessageQueue:
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {}
        self.handlers: Dict[str, Callable] = {}
    
    def create_queue(self, queue_name: str):
        if queue_name not in self.queues:
            self.queues[queue_name] = asyncio.Queue()
    
    def register_handler(self, queue_name: str, handler: Callable):
        self.handlers[queue_name] = handler
    
    async def publish(self, queue_name: str, message: Dict[str, Any]):
        if queue_name not in self.queues:
            self.create_queue(queue_name)
        
        await self.queues[queue_name].put(message)
    
    async def consume(self, queue_name: str):
        if queue_name not in self.queues:
            raise ValueError(f"Queue {queue_name} does not exist")
        
        handler = self.handlers.get(queue_name)
        if not handler:
            raise ValueError(f"No handler for queue {queue_name}")
        
        while True:
            message = await self.queues[queue_name].get()
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
            finally:
                self.queues[queue_name].task_done()

# Usage
queue = MessageQueue()

async def handle_user_created(message: Dict[str, Any]):
    user_id = message["user_id"]
    await send_welcome_email(user_id)
    await create_user_profile(user_id)

queue.register_handler("user_created", handle_user_created)

# Start consumer
asyncio.create_task(queue.consume("user_created"))

# Publish message
await queue.publish("user_created", {"user_id": "123", "email": "user@example.com"})
```

### **Q108: How do you implement distributed locks?**
```python
import asyncio
import time
import uuid

class DistributedLock:
    def __init__(self, redis_client, lock_name: str, timeout: int = 10):
        self.redis = redis_client
        self.lock_name = f"lock:{lock_name}"
        self.timeout = timeout
        self.identifier = str(uuid.uuid4())
    
    async def acquire(self) -> bool:
        end_time = time.time() + self.timeout
        
        while time.time() < end_time:
            if await self.redis.set(self.lock_name, self.identifier, nx=True, ex=self.timeout):
                return True
            await asyncio.sleep(0.01)
        
        return False
    
    async def release(self):
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        await self.redis.eval(script, 1, self.lock_name, self.identifier)
    
    async def __aenter__(self):
        if not await self.acquire():
            raise Exception("Could not acquire lock")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()

# Usage
async def critical_section():
    async with DistributedLock(redis_client, "user_update_123"):
        # Critical section - only one process can execute this
        await update_user_data("123")
```

### **Q109: How do you implement load balancing strategies?**
```python
import random
import time
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Server:
    host: str
    port: int
    weight: int = 1
    active_connections: int = 0
    response_time: float = 0.0
    is_healthy: bool = True

class LoadBalancer:
    def __init__(self, servers: List[Server]):
        self.servers = servers
    
    def round_robin(self) -> Server:
        healthy_servers = [s for s in self.servers if s.is_healthy]
        if not healthy_servers:
            raise Exception("No healthy servers available")
        
        # Simple round-robin implementation
        server = healthy_servers[0]
        self.servers.remove(server)
        self.servers.append(server)
        return server
    
    def weighted_round_robin(self) -> Server:
        healthy_servers = [s for s in self.servers if s.is_healthy]
        if not healthy_servers:
            raise Exception("No healthy servers available")
        
        total_weight = sum(s.weight for s in healthy_servers)
        random_weight = random.randint(1, total_weight)
        
        current_weight = 0
        for server in healthy_servers:
            current_weight += server.weight
            if random_weight <= current_weight:
                return server
    
    def least_connections(self) -> Server:
        healthy_servers = [s for s in self.servers if s.is_healthy]
        if not healthy_servers:
            raise Exception("No healthy servers available")
        
        return min(healthy_servers, key=lambda s: s.active_connections)
    
    def fastest_response(self) -> Server:
        healthy_servers = [s for s in self.servers if s.is_healthy and s.response_time > 0]
        if not healthy_servers:
            return self.round_robin()
        
        return min(healthy_servers, key=lambda s: s.response_time)
```

### **Q110: How do you implement health checks for distributed systems?**
```python
import asyncio
import time
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    response_time: float
    error_message: str = None
    last_check: float = None

class HealthChecker:
    def __init__(self):
        self.checks: List[Callable] = []
        self.results: Dict[str, HealthCheck] = {}
    
    def register_check(self, name: str, check_func: Callable):
        self.checks.append((name, check_func))
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        tasks = []
        for name, check_func in self.checks:
            task = asyncio.create_task(self._run_single_check(name, check_func))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            name, _ = self.checks[i]
            if isinstance(result, Exception):
                self.results[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    response_time=0.0,
                    error_message=str(result),
                    last_check=time.time()
                )
            else:
                self.results[name] = result
        
        return self.results
    
    async def _run_single_check(self, name: str, check_func: Callable) -> HealthCheck:
        start_time = time.time()
        try:
            result = await check_func()
            response_time = time.time() - start_time
            
            return HealthCheck(
                name=name,
                status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                response_time=response_time,
                last_check=time.time()
            )
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                error_message=str(e),
                last_check=time.time()
            )

# Usage
health_checker = HealthChecker()

async def check_database():
    # Check database connection
    return True

async def check_redis():
    # Check Redis connection
    return True

async def check_external_api():
    # Check external API
    return True

health_checker.register_check("database", check_database)
health_checker.register_check("redis", check_redis)
health_checker.register_check("external_api", check_external_api)

@app.get("/health")
async def health_endpoint():
    results = await health_checker.run_all_checks()
    
    overall_status = HealthStatus.HEALTHY
    for check in results.values():
        if check.status == HealthStatus.UNHEALTHY:
            overall_status = HealthStatus.UNHEALTHY
            break
        elif check.status == HealthStatus.DEGRADED:
            overall_status = HealthStatus.DEGRADED
    
    return {
        "status": overall_status.value,
        "checks": {name: {
            "status": check.status.value,
            "response_time": check.response_time,
            "error": check.error_message
        } for name, check in results.items()}
    }
```

## **Q111-Q140: TROUBLESHOOTING & DEBUGGING**

### **Q111: How do you debug async/await issues?**
```python
import asyncio
import logging
import traceback

# Enable asyncio debug mode
asyncio.get_event_loop().set_debug(True)

# Custom async debugger
class AsyncDebugger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def debug_async_function(self, func, *args, **kwargs):
        try:
            self.logger.info(f"Starting async function: {func.__name__}")
            result = await func(*args, **kwargs)
            self.logger.info(f"Completed async function: {func.__name__}")
            return result
        except Exception as e:
            self.logger.error(f"Error in async function {func.__name__}: {e}")
            self.logger.error(traceback.format_exc())
            raise

# Usage
debugger = AsyncDebugger()

async def problematic_function():
    await asyncio.sleep(1)
    raise ValueError("Something went wrong")

# Debug the function
result = await debugger.debug_async_function(problematic_function)
```

### **Q112: How do you debug database connection issues?**
```python
import logging
from sqlalchemy import event
from sqlalchemy.engine import Engine

# Enable SQL logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Add connection event listeners
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

@event.listens_for(Engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    logging.info(f"Executing: {statement}")
    logging.info(f"Parameters: {parameters}")

# Database health check
async def debug_database_connection(db: AsyncSession):
    try:
        # Test basic connection
        result = await db.execute(text("SELECT 1"))
        logging.info("Database connection successful")
        
        # Test transaction
        await db.begin()
        await db.rollback()
        logging.info("Database transaction test successful")
        
        # Check connection pool
        pool = db.bind.pool
        logging.info(f"Pool size: {pool.size()}")
        logging.info(f"Checked out connections: {pool.checkedout()}")
        
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        raise
```

### **Q113: How do you debug memory leaks?**
```python
import tracemalloc
import gc
import psutil
import os

class MemoryProfiler:
    def __init__(self):
        tracemalloc.start()
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self):
        memory_info = self.process.memory_info()
        return {
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
        }
    
    def get_tracemalloc_snapshot(self):
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        return [
            {
                "filename": stat.traceback.format()[0],
                "size": stat.size / 1024 / 1024,  # MB
                "count": stat.count
            }
            for stat in top_stats[:10]
        ]
    
    def force_garbage_collection(self):
        collected = gc.collect()
        return collected

# Usage
profiler = MemoryProfiler()

@app.middleware("http")
async def memory_profiling_middleware(request: Request, call_next):
    start_memory = profiler.get_memory_usage()
    
    response = await call_next(request)
    
    end_memory = profiler.get_memory_usage()
    memory_diff = end_memory["rss"] - start_memory["rss"]
    
    if memory_diff > 10:  # More than 10MB increase
        logging.warning(f"Memory usage increased by {memory_diff:.2f}MB")
        top_stats = profiler.get_tracemalloc_snapshot()
        logging.warning(f"Top memory allocations: {top_stats}")
    
    return response
```

### **Q114: How do you debug performance bottlenecks?**
```python
import time
import cProfile
import pstats
from functools import wraps

class PerformanceProfiler:
    def __init__(self):
        self.profiler = cProfile.Profile()
    
    def profile_function(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Profile the function
            self.profiler.enable()
            try:
                result = await func(*args, **kwargs)
            finally:
                self.profiler.disable()
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            logging.info(f"Function {func.__name__} took {execution_time:.4f} seconds")
            
            return result
        return wrapper
    
    def get_profile_stats(self):
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        return stats

# Usage
profiler = PerformanceProfiler()

@profiler.profile_function
async def slow_function():
    await asyncio.sleep(1)
    return "done"

# Custom timing decorator
def time_function(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        logging.info(f"{func.__name__} executed in {end - start:.4f} seconds")
        return result
    return wrapper
```

### **Q115: How do you debug authentication issues?**
```python
import jwt
from datetime import datetime

class AuthDebugger:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def decode_token_debug(self, token: str):
        try:
            # Decode without verification first
            unverified = jwt.decode(token, options={"verify_signature": False})
            logging.info(f"Token payload (unverified): {unverified}")
            
            # Check expiration
            exp = unverified.get('exp')
            if exp:
                exp_time = datetime.fromtimestamp(exp)
                now = datetime.now()
                if exp_time < now:
                    logging.error(f"Token expired at {exp_time}, current time: {now}")
                else:
                    logging.info(f"Token expires at {exp_time}")
            
            # Now decode with verification
            verified = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            logging.info(f"Token payload (verified): {verified}")
            return verified
            
        except jwt.ExpiredSignatureError:
            logging.error("Token has expired")
            raise
        except jwt.InvalidTokenError as e:
            logging.error(f"Invalid token: {e}")
            raise
    
    def validate_token_structure(self, token: str):
        parts = token.split('.')
        if len(parts) != 3:
            logging.error(f"Token should have 3 parts, got {len(parts)}")
            return False
        
        try:
            header = jwt.get_unverified_header(token)
            logging.info(f"Token header: {header}")
            return True
        except Exception as e:
            logging.error(f"Invalid token structure: {e}")
            return False
```

### **Q116: How do you debug API response issues?**
```python
from fastapi import Request, Response
import json

class APIResponseDebugger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def debug_request_response(self, request: Request, response: Response):
        # Log request details
        self.logger.info(f"Request: {request.method} {request.url}")
        self.logger.info(f"Headers: {dict(request.headers)}")
        
        # Log request body if present
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    self.logger.info(f"Request body: {body.decode()}")
            except Exception as e:
                self.logger.error(f"Error reading request body: {e}")
        
        # Log response details
        self.logger.info(f"Response status: {response.status_code}")
        self.logger.info(f"Response headers: {dict(response.headers)}")
        
        # Log response body if it's JSON
        if response.headers.get("content-type", "").startswith("application/json"):
            try:
                response_body = response.body
                if response_body:
                    json_data = json.loads(response_body.decode())
                    self.logger.info(f"Response body: {json.dumps(json_data, indent=2)}")
            except Exception as e:
                self.logger.error(f"Error parsing response body: {e}")

# Middleware to debug all requests
@app.middleware("http")
async def debug_middleware(request: Request, call_next):
    debugger = APIResponseDebugger()
    
    # Store original response
    response = await call_next(request)
    
    # Debug the request/response
    await debugger.debug_request_response(request, response)
    
    return response
```

### **Q117: How do you debug WebSocket connection issues?**
```python
import websockets
from websockets.exceptions import ConnectionClosed

class WebSocketDebugger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connections = {}
    
    async def debug_websocket_connection(self, websocket: WebSocket, client_id: str):
        self.connections[client_id] = {
            "connected_at": datetime.now(),
            "messages_sent": 0,
            "messages_received": 0,
            "last_activity": datetime.now()
        }
        
        self.logger.info(f"WebSocket connection established: {client_id}")
        
        try:
            while True:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    self.connections[client_id]["messages_received"] += 1
                    self.connections[client_id]["last_activity"] = datetime.now()
                    
                    self.logger.info(f"Received from {client_id}: {message}")
                    
                    # Echo back the message
                    await websocket.send_text(f"Echo: {message}")
                    self.connections[client_id]["messages_sent"] += 1
                    
                except asyncio.TimeoutError:
                    # Send ping to check if connection is alive
                    await websocket.ping()
                    self.logger.info(f"Sent ping to {client_id}")
                    
                except ConnectionClosed:
                    self.logger.info(f"WebSocket connection closed: {client_id}")
                    break
                    
        except Exception as e:
            self.logger.error(f"WebSocket error for {client_id}: {e}")
        finally:
            if client_id in self.connections:
                del self.connections[client_id]
    
    def get_connection_stats(self):
        return {
            client_id: {
                "duration": (datetime.now() - conn["connected_at"]).total_seconds(),
                "messages_sent": conn["messages_sent"],
                "messages_received": conn["messages_received"],
                "last_activity": conn["last_activity"]
            }
            for client_id, conn in self.connections.items()
        }
```

### **Q118: How do you debug caching issues?**
```python
class CacheDebugger:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.logger = logging.getLogger(__name__)
    
    async def debug_cache_operation(self, operation: str, key: str, value=None, ttl=None):
        self.logger.info(f"Cache {operation}: key={key}")
        
        if operation == "get":
            result = await self.redis.get(key)
            self.logger.info(f"Cache GET result: {result}")
            return result
            
        elif operation == "set":
            await self.redis.setex(key, ttl or 3600, value)
            self.logger.info(f"Cache SET: key={key}, value={value}, ttl={ttl}")
            
        elif operation == "delete":
            result = await self.redis.delete(key)
            self.logger.info(f"Cache DELETE result: {result}")
            return result
    
    async def check_cache_health(self):
        try:
            # Test basic operations
            test_key = "health_check"
            test_value = "test_value"
            
            await self.redis.setex(test_key, 60, test_value)
            retrieved = await self.redis.get(test_key)
            await self.redis.delete(test_key)
            
            if retrieved == test_value:
                self.logger.info("Cache health check passed")
                return True
            else:
                self.logger.error("Cache health check failed: value mismatch")
                return False
                
        except Exception as e:
            self.logger.error(f"Cache health check failed: {e}")
            return False
    
    async def get_cache_stats(self):
        info = await self.redis.info()
        return {
            "connected_clients": info.get("connected_clients"),
            "used_memory": info.get("used_memory_human"),
            "keyspace_hits": info.get("keyspace_hits"),
            "keyspace_misses": info.get("keyspace_misses"),
            "hit_rate": info.get("keyspace_hits", 0) / (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1))
        }
```

### **Q119: How do you debug background task issues?**
```python
import asyncio
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TaskInfo:
    task_id: str
    name: str
    status: str
    created_at: datetime
    started_at: datetime = None
    completed_at: datetime = None
    error: str = None

class BackgroundTaskDebugger:
    def __init__(self):
        self.tasks: Dict[str, TaskInfo] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_task(self, task_id: str, name: str):
        self.tasks[task_id] = TaskInfo(
            task_id=task_id,
            name=name,
            status="pending",
            created_at=datetime.now()
        )
        self.logger.info(f"Registered background task: {task_id} - {name}")
    
    def start_task(self, task_id: str):
        if task_id in self.tasks:
            self.tasks[task_id].status = "running"
            self.tasks[task_id].started_at = datetime.now()
            self.logger.info(f"Started background task: {task_id}")
    
    def complete_task(self, task_id: str, success: bool = True, error: str = None):
        if task_id in self.tasks:
            self.tasks[task_id].status = "completed" if success else "failed"
            self.tasks[task_id].completed_at = datetime.now()
            if error:
                self.tasks[task_id].error = error
            self.logger.info(f"Completed background task: {task_id} - {self.tasks[task_id].status}")
    
    def get_task_stats(self):
        stats = {
            "total": len(self.tasks),
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0
        }
        
        for task in self.tasks.values():
            stats[task.status] += 1
        
        return stats
    
    def get_running_tasks(self):
        return [task for task in self.tasks.values() if task.status == "running"]
    
    def get_failed_tasks(self):
        return [task for task in self.tasks.values() if task.status == "failed"]

# Usage
task_debugger = BackgroundTaskDebugger()

async def background_task_with_debugging(task_id: str, task_name: str):
    task_debugger.register_task(task_id, task_name)
    task_debugger.start_task(task_id)
    
    try:
        # Your background task logic here
        await asyncio.sleep(5)  # Simulate work
        task_debugger.complete_task(task_id, success=True)
    except Exception as e:
        task_debugger.complete_task(task_id, success=False, error=str(e))
        raise
```

### **Q120: How do you debug database query performance?**
```python
import time
from sqlalchemy import event
from sqlalchemy.engine import Engine

class QueryPerformanceDebugger:
    def __init__(self):
        self.queries = []
        self.logger = logging.getLogger(__name__)
    
    def setup_query_logging(self, engine: Engine):
        @event.listens_for(engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()
        
        @event.listens_for(engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total = time.time() - context._query_start_time
            
            query_info = {
                "statement": statement,
                "parameters": parameters,
                "duration": total,
                "timestamp": datetime.now()
            }
            
            self.queries.append(query_info)
            
            if total > 1.0:  # Log slow queries
                self.logger.warning(f"Slow query detected ({total:.2f}s): {statement}")
    
    def get_slow_queries(self, threshold: float = 1.0):
        return [q for q in self.queries if q["duration"] > threshold]
    
    def get_query_stats(self):
        if not self.queries:
            return {"total_queries": 0}
        
        durations = [q["duration"] for q in self.queries]
        return {
            "total_queries": len(self.queries),
            "average_duration": sum(durations) / len(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "slow_queries": len(self.get_slow_queries())
        }
    
    def analyze_query_patterns(self):
        patterns = {}
        for query in self.queries:
            # Extract table name from query
            statement = query["statement"].lower()
            if "from" in statement:
                table = statement.split("from")[1].split()[0]
                if table not in patterns:
                    patterns[table] = {"count": 0, "total_time": 0}
                patterns[table]["count"] += 1
                patterns[table]["total_time"] += query["duration"]
        
        return patterns

# Usage
query_debugger = QueryPerformanceDebugger()
query_debugger.setup_query_logging(engine)

@app.get("/debug/queries")
async def get_query_debug_info():
    return {
        "stats": query_debugger.get_query_stats(),
        "slow_queries": query_debugger.get_slow_queries(),
        "patterns": query_debugger.analyze_query_patterns()
    }
```

## **Q121-Q150: CODE REVIEW & BEST PRACTICES**

### **Q121: What are FastAPI best practices for error handling?**
```python
# Custom exception classes
class BusinessLogicError(Exception):
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

# Global exception handlers
@app.exception_handler(BusinessLogicError)
async def business_logic_exception_handler(request: Request, exc: BusinessLogicError):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Business Logic Error",
            "message": exc.message,
            "error_code": exc.error_code,
            "path": str(request.url)
        }
    )

# Consistent error responses
class ErrorResponse(BaseModel):
    error: str
    message: str
    error_code: str = None
    details: Dict[str, Any] = None
    timestamp: datetime = Field(default_factory=datetime.now)
```

### **Q122: What are FastAPI best practices for validation?**
```python
# Use Pydantic for request validation
class UserCreate(BaseModel):
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")
    full_name: str = Field(..., min_length=2, max_length=100)
    
    @validator('password')
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        return v
    
    @validator('email')
    def validate_email_domain(cls, v):
        allowed_domains = ['gmail.com', 'yahoo.com', 'company.com']
        domain = v.split('@')[1]
        if domain not in allowed_domains:
            raise ValueError('Email domain not allowed')
        return v

# Use response models for output validation
@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate):
    # Implementation
    pass
```

### **Q123: What are FastAPI best practices for security?**
```python
# Use HTTPS in production
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
app.add_middleware(HTTPSRedirectMiddleware)

# Implement proper CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Specific methods only
    allow_headers=["*"],
)

# Use secure headers
from fastapi.middleware.trustedhost import TrustedHostMiddleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["yourdomain.com"])

# Implement rate limiting
from slo
