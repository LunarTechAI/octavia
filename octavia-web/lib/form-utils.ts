// lib/form-utils.ts - Helper utilities for FormData
export function createFormData(data: Record<string, any>): FormData {
  const formData = new FormData();
  
  Object.entries(data).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      if (value instanceof File) {
        formData.append(key, value);
      } else if (Array.isArray(value)) {
        value.forEach(item => {
          formData.append(key, String(item));
        });
      } else {
        formData.append(key, String(value));
      }
    }
  });
  
  return formData;
}

export function createAuthFormData(email: string, password: string, name?: string): FormData {
  const formData = new FormData();
  formData.append('email', email);
  formData.append('password', password);
  if (name) {
    formData.append('name', name);
  }
  return formData;
}

export function createVideoUploadFormData(
  file: File,
  userEmail: string,
  targetLanguage: string = 'es',
  chunkSize?: number
): FormData {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('user_email', userEmail);
  formData.append('target_language', targetLanguage);
  if (chunkSize !== undefined) {
    formData.append('chunk_size', chunkSize.toString());
  }
  return formData;
}