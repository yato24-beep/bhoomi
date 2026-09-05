import io


def test_register_user_success(client):
    """Test registering a new user."""
    payload = {
        "email": "new_user@docplatform.com",
        "password": "strongpassword123",
        "full_name": "New Employee",
        "role": "OFFICER",
    }
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "new_user@docplatform.com"
    assert data["role"] == "OFFICER"
    assert data["full_name"] == "New Employee"
    assert "id" in data
    assert "password" not in data


def test_register_duplicate_user(client):
    """Test registering an existing email returns 400 Bad Request."""
    payload = {
        "email": "duplicate@docplatform.com",
        "password": "password123",
        "role": "VIEWER",
    }
    res1 = client.post("/api/v1/auth/register", json=payload)
    assert res1.status_code == 201

    res2 = client.post("/api/v1/auth/register", json=payload)
    assert res2.status_code == 400
    assert "already exists" in res2.json()["detail"]


def test_login_and_me_profile(client):
    """Test login with credentials and accessing /auth/me profile."""
    # 1. Register
    client.post(
        "/api/v1/auth/register",
        json={"email": "login_test@doc.com", "password": "mypassword123", "role": "REVIEWER"},
    )

    # 2. Login with valid password
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": "login_test@doc.com", "password": "mypassword123"},
    )
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    assert token_data["user"]["role"] == "REVIEWER"

    token = token_data["access_token"]

    # 3. Access /auth/me with Bearer token
    me_res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["email"] == "login_test@doc.com"
    assert me_data["role"] == "REVIEWER"


def test_login_invalid_password(client):
    """Test login with invalid password returns 401."""
    client.post(
        "/api/v1/auth/register",
        json={"email": "wrong_pwd@doc.com", "password": "correct_password", "role": "VIEWER"},
    )
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "wrong_pwd@doc.com", "password": "incorrect_password"},
    )
    assert response.status_code == 401
    assert "Incorrect email or password" in response.json()["detail"]


def test_rbac_upload_permissions(client, viewer_headers, officer_headers):
    """Verify VIEWER cannot upload (403), but OFFICER can upload (201)."""
    files = {"file": ("rbac_test.pdf", io.BytesIO(b"%PDF-1.4 rbac test"), "application/pdf")}

    # VIEWER upload attempt -> 403 Forbidden
    viewer_res = client.post("/api/v1/documents/upload", files=files, headers=viewer_headers)
    assert viewer_res.status_code == 403
    assert "Permission denied" in viewer_res.json()["detail"]

    # OFFICER upload attempt -> 201 Created
    files_officer = {"file": ("rbac_test.pdf", io.BytesIO(b"%PDF-1.4 rbac test"), "application/pdf")}
    officer_res = client.post("/api/v1/documents/upload", files=files_officer, headers=officer_headers)
    assert officer_res.status_code == 201


def test_rbac_delete_permissions(client, officer_headers, admin_headers):
    """Verify OFFICER cannot delete (403), but ADMIN can delete (204)."""
    # Create doc
    files = {"file": ("to_delete_rbac.pdf", io.BytesIO(b"%PDF-1.4 delete test"), "application/pdf")}
    create_res = client.post("/api/v1/documents/upload", files=files, headers=officer_headers)
    assert create_res.status_code == 201
    doc_id = create_res.json()["document"]["id"]

    # OFFICER delete attempt -> 403 Forbidden
    officer_del = client.delete(f"/api/v1/documents/{doc_id}", headers=officer_headers)
    assert officer_del.status_code == 403
    assert "Permission denied" in officer_del.json()["detail"]

    # ADMIN delete attempt -> 204 No Content
    admin_del = client.delete(f"/api/v1/documents/{doc_id}", headers=admin_headers)
    assert admin_del.status_code == 204
