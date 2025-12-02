---
name: api-documenter
description: Creates comprehensive API documentation from code, including endpoints, parameters, responses, and examples. Use when documenting REST APIs, GraphQL APIs, or generating OpenAPI/Swagger specifications.
version: 1.0.0
tags: [development, documentation, api, openapi]
---

# API Documenter

Generate clear, comprehensive API documentation that developers will love to use.

## Instructions

When creating API documentation:

1. **Analyze the API**: Understand endpoints, methods, and data structures
2. **Document Systematically**: Follow a consistent structure for each endpoint
3. **Include Examples**: Provide request and response examples
4. **Explain Clearly**: Use clear language, avoid jargon
5. **Cover Edge Cases**: Document error responses and special cases
6. **Keep Updated**: Ensure documentation matches current implementation

## Documentation Structure

### API Overview
- Base URL
- Authentication method
- Rate limiting
- Common headers
- API versioning

### Endpoint Documentation

For each endpoint, include:

#### Endpoint Header
```
METHOD /path/to/endpoint
```

#### Description
Brief explanation of what the endpoint does.

#### Authentication
Required authentication level or method.

#### Request Parameters

**Path Parameters**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| id   | integer | Yes | User ID |

**Query Parameters**
| Name | Type | Required | Description | Default |
|------|------|----------|-------------|---------|
| limit | integer | No | Results per page | 10 |
| offset | integer | No | Results offset | 0 |

**Request Body** (for POST/PUT/PATCH)
```json
{
  "field": "type - description",
  "required_field": "string - required description"
}
```

#### Response

**Success Response (200 OK)**
```json
{
  "status": "success",
  "data": {
    "id": 1,
    "name": "John Doe"
  }
}
```

**Error Responses**
- `400 Bad Request` - Invalid input
- `401 Unauthorized` - Authentication required
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

#### Example Request

```bash
curl -X GET "https://api.example.com/v1/users/123" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json"
```

```javascript
// JavaScript
fetch('https://api.example.com/v1/users/123', {
  headers: {
    'Authorization': 'Bearer YOUR_TOKEN'
  }
})
```

```python
# Python
import requests
response = requests.get(
    'https://api.example.com/v1/users/123',
    headers={'Authorization': 'Bearer YOUR_TOKEN'}
)
```

#### Example Response

```json
{
  "status": "success",
  "data": {
    "id": 123,
    "name": "John Doe",
    "email": "john@example.com",
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

## Example Documentation

### Complete Endpoint Example

```markdown
## Get User by ID

GET /v1/users/{id}

Retrieves a single user by their unique identifier.

### Authentication
Requires Bearer token authentication.

### Path Parameters
| Name | Type | Required | Description |
|------|------|----------|-------------|
| id   | integer | Yes | Unique user identifier |

### Response

**200 OK**
\`\`\`json
{
  "id": 123,
  "name": "John Doe",
  "email": "john@example.com",
  "role": "admin",
  "created_at": "2024-01-01T00:00:00Z"
}
\`\`\`

**404 Not Found**
\`\`\`json
{
  "error": "User not found",
  "code": "USER_NOT_FOUND"
}
\`\`\`

### Example Request
\`\`\`bash
curl -X GET "https://api.example.com/v1/users/123" \\
  -H "Authorization: Bearer YOUR_TOKEN"
\`\`\`
```

## Guidelines

### Be Comprehensive
- Document all endpoints
- Include all parameters
- List all possible responses
- Provide multiple examples

### Be Clear
- Use simple, direct language
- Define technical terms
- Include context and purpose
- Explain business logic when relevant

### Be Consistent
- Use same format for all endpoints
- Consistent naming conventions
- Standard response structures
- Uniform example style

### Be Practical
- Real-world examples
- Common use cases
- Error handling patterns
- Best practices

## OpenAPI/Swagger Generation

When generating OpenAPI 3.0 specification:

```yaml
openapi: 3.0.0
info:
  title: API Name
  version: 1.0.0
  description: API description
servers:
  - url: https://api.example.com/v1
paths:
  /users/{id}:
    get:
      summary: Get user by ID
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: integer
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: integer
        name:
          type: string
```

## Common Sections to Include

### Getting Started
- Quick start guide
- Authentication setup
- First API call
- Common patterns

### Reference
- All endpoints
- Data models
- Error codes
- Rate limits

### Guides
- Authentication guide
- Pagination guide
- Filtering guide
- Webhooks guide

### Resources
- SDKs and libraries
- Postman collection
- Code examples
- Support information

## Tips for Great API Docs

- **Test Your Examples**: Ensure all code examples actually work
- **Version Changes**: Document what changed between versions
- **Search Friendly**: Use clear headings and keywords
- **Interactive**: Consider tools like Swagger UI or Postman
- **Keep Updated**: Document changes immediately
- **Get Feedback**: Ask developers what's unclear
- **Show Errors**: Don't just show happy path
- **Performance Notes**: Document timeouts, limits, best practices

## Research-Backed Best Practices

### The Evolution from REST to GraphQL
Recent research (Addanki, 2025; Zanevych, 2024) reveals important trends in API design:

**REST remains dominant for public APIs** but faces challenges:
- **Over-fetching/under-fetching**: Clients receive too much or too little data
- **Multiple round trips**: Complex queries require multiple endpoint calls
- **Rigid structure**: Predetermined endpoints limit flexibility

**GraphQL addresses REST limitations**:
- **Schema-driven approach**: Clients specify exactly what data they need
- **Single request**: Complex data requirements satisfied in one call
- **Improved developer experience**: Self-documenting through introspection
- **Growing adoption**: Becoming market leader for dynamic, data-heavy applications

**Practical implications**:
- REST for simple, cacheable, public-facing APIs
- GraphQL for complex, dynamic applications with varied client needs
- Consider hybrid approaches for existing systems

### API Documentation and Developer Experience
Research on bioinformatics APIs (Ireland & Martin, 2021) demonstrates GraphQL benefits:
- **Self-documenting nature**: GraphQL schemas make APIs explorable without external docs
- **Complex relationships**: Graph structure naturally represents interconnected data
- **Single query flexibility**: Any hierarchy or nesting level accessible in one request
- **Better than REST for complex domains**: Reduces need for extensive documentation

**Key Insight**: Modern API documentation should embrace interactive exploration. GraphQL's introspection and tools like GraphQL Playground reduce documentation burden while improving developer experience.

### Performance Considerations
Comparative studies (Seabra et al., 2019; Wittern et al., 2017) show:
- **REST advantage**: Better caching, mature tooling, predictable performance
- **GraphQL advantage**: Reduced network overhead for complex queries, flexible data fetching
- **Trade-off**: GraphQL queries can be more expensive server-side but reduce round trips

**Documentation Recommendation**: Always document query complexity, rate limits, and performance characteristics specific to your API design.

## References

- Addanki, S. (2025). "Architectural Shifts in API Design: The Progression from SOAP to REST and GraphQL". International Journal on Science and Technology, 16(2).
- Zanevych, O. (2024). "Advancing Web Development: A Comparative Analysis of Modern Frameworks for REST and GraphQL Back-End Services". Grail of Science, 37, 216-228.
- Ireland, S.M., & Martin, A.C.R. (2021). "GraphQL for the delivery of bioinformatics web APIs and application to ZincBind". Bioinformatics Advances, 1(1).
- Seabra, M., Nazário, M.F., & Pinto, G. (2019). "REST or GraphQL? A performance comparative study". Proceedings of the XIII Brazilian Symposium on Software Components, Architectures, and Reuse.
