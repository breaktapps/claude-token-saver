interface RequestConfig {
  timeout: number;
  retries: number;
  baseUrl: string;
}

interface ApiResponse<T> {
  data: T;
  status: number;
  message: string;
}

export function createClient(config: RequestConfig): ApiClient {
  const client = new ApiClient(config);
  client.initialize();
  return client;
}

export function handleError(error: Error): ApiResponse<null> {
  logError(error);
  return {
    data: null,
    status: 500,
    message: error.message,
  };
}

class ApiClient {
  private config: RequestConfig;

  constructor(config: RequestConfig) {
    this.config = config;
  }

  initialize(): void {
    validateConfig(this.config);
  }

  async fetch<T>(endpoint: string): Promise<ApiResponse<T>> {
    const url = buildUrl(this.config.baseUrl, endpoint);
    const response = await makeRequest(url, this.config);
    return parseResponse(response);
  }
}
